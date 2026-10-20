"""Per-run resource measurements and actual minibatch-objective convergence curves."""

import csv
from datetime import datetime, timezone
import json
import math
import threading
import time

import numpy as np
import psutil
import torch

from datautils.partition_store import atomic_output
from experiments.paths import project_path


MIB = 1024 ** 2
CURVE_FIELDS = (
    "round", "train_loss", "train_steps", "nonfinite_loss_steps", "participating_clients",
    "labeled_train_loss", "unlabeled_train_loss", "mixed_train_loss",
    "training_time_seconds", "cumulative_training_time_seconds", "elapsed_time_seconds",
    "cpu_peak_rss_mib", "cpu_tree_peak_rss_mib", "cuda_peak_allocated_mib", "cuda_peak_reserved_mib",
)


def record_training_loss(client, loss):
    recorder = getattr(client, "training_recorder", None)
    if recorder is not None:
        recorder.record_loss(client.id_user, client.tag_client_lbulb, loss)


class TrainingRecorder:
    """Measure one run; RSS peaks are sampled, CUDA peaks use allocator counters."""

    def __init__(self, args, device, run_number, seed, sample_interval=0.1):
        self.args = args
        self.device = torch.device(device)
        self.run_number = run_number
        self.seed = seed
        self.sample_interval = sample_interval
        directory = project_path(args.metrics_dir) / args.dataset / args.save_name
        self.filename = directory / "run_{}_training.json".format(run_number)
        self.rounds = []
        self.warmup = []
        self.interrupted_round = None
        self.status = "running"
        self.error_type = None
        self.measurement_errors = set()
        self.started_at = None
        self.finished_at = None
        self.total_time_seconds = None
        self.training_time_seconds = 0.0
        self._active_round = None
        self._active_warmup = False
        self._losses = {}
        self._clients = set()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._process = psutil.Process()
        self._peak_rss = 0
        self._peak_tree_rss = 0
        self._round_peak_rss = 0
        self._round_peak_tree_rss = 0
        self._samples = 0
        self._cuda_allocated = None
        self._cuda_reserved = None

    def __enter__(self):
        self._synchronize()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        self.started_at = datetime.now(timezone.utc).isoformat()
        self._start_time = time.perf_counter()
        self._sample_memory()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()
        try:
            self.save()
        except BaseException:
            self._stop.set()
            self._thread.join()
            raise
        return self

    def __exit__(self, exception_type, exception, traceback):
        self.status = ("completed" if exception_type is None else
                       "interrupted" if issubclass(exception_type, KeyboardInterrupt) else "failed")
        self.error_type = exception_type.__name__ if exception_type else None
        try:
            try:
                self._synchronize()
            except Exception as error:
                self.measurement_errors.add("Final CUDA synchronization: " + type(error).__name__)
            if self._active_round is not None:
                try:
                    self.interrupted_round = self._finish_round(completed=False)
                except Exception as error:
                    self.interrupted_round = {"round": self._active_round,
                                              "measurement_error": type(error).__name__}
                    with self._lock:
                        self._active_round = None
                    self._losses = {}
                    self.measurement_errors.add("Interrupted round measurement: " + type(error).__name__)
            self._capture_cuda_peaks()
            self._sample_memory()
            self.total_time_seconds = time.perf_counter() - self._start_time
            self.finished_at = datetime.now(timezone.utc).isoformat()
            if self.status == "completed" and (self.interrupted_round is not None or
                                                len(self.rounds) != self.args.num_global_iters):
                self.status = "incomplete"
        finally:
            self._stop.set()
            if self._thread is not None:
                self._thread.join()
        try:
            self.save()
        except Exception:
            if exception_type is None:
                raise
        return False

    def attach(self, server):
        server.training_recorder = self
        for client in server.users:
            client.training_recorder = self

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _sample_loop(self):
        while not self._stop.wait(self.sample_interval):
            self._sample_memory()

    def _sample_memory(self):
        try:
            rss = self._process.memory_info().rss
            tree_rss = rss
            for child in self._process.children(recursive=True):
                try:
                    tree_rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.ZombieProcess):
                    continue
                except psutil.AccessDenied:
                    with self._lock:
                        self.measurement_errors.add("Some child RSS measurements were denied")
            with self._lock:
                self._peak_rss = max(self._peak_rss, rss)
                self._peak_tree_rss = max(self._peak_tree_rss, tree_rss)
                if self._active_round is not None:
                    self._round_peak_rss = max(self._round_peak_rss, rss)
                    self._round_peak_tree_rss = max(self._round_peak_tree_rss, tree_rss)
                self._samples += 1
        except psutil.Error as error:
            with self._lock:
                self.measurement_errors.add("RSS sampling: " + type(error).__name__)

    def _capture_cuda_peaks(self):
        if self.device.type != "cuda":
            return None, None
        try:
            allocated = torch.cuda.max_memory_allocated(self.device) / MIB
            reserved = torch.cuda.max_memory_reserved(self.device) / MIB
            self._cuda_allocated = max(self._cuda_allocated or 0, allocated)
            self._cuda_reserved = max(self._cuda_reserved or 0, reserved)
            return allocated, reserved
        except Exception as error:
            self.measurement_errors.add("CUDA memory: " + type(error).__name__)
            return None, None

    def begin_round(self, round_number, warmup=False):
        if self._active_round is not None:
            raise RuntimeError("Previous training round has not ended")
        self._synchronize()
        self._capture_cuda_peaks()
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        with self._lock:
            self._round_peak_rss = 0
            self._round_peak_tree_rss = 0
            self._active_round = round_number
            self._active_warmup = warmup
        self._losses = {}
        self._clients = set()
        self._round_start = time.perf_counter()
        self._sample_memory()

    @torch.no_grad()
    def record_loss(self, client_id, tag, loss):
        if self._active_round is None:
            raise RuntimeError("Cannot record a training loss outside a training round")
        value = loss.detach().float().reshape(())
        key = (int(tag), str(value.device))
        finite = torch.isfinite(value)
        contribution = torch.stack((torch.where(finite, value, torch.zeros_like(value)), (~finite).float()))
        if key not in self._losses:
            self._losses[key] = [contribution, 1]
        else:
            self._losses[key][0].add_(contribution)
            self._losses[key][1] += 1
        self._clients.add(int(client_id))

    def end_round(self):
        self._synchronize()
        record = self._finish_round(completed=True)
        self.save()
        return record

    def _finish_round(self, completed):
        if self._active_round is None:
            raise RuntimeError("No active training round")
        duration = time.perf_counter() - self._round_start
        self.training_time_seconds += duration
        self._sample_memory()
        allocated, reserved = self._capture_cuda_peaks()
        totals = {tag: [0.0, 0, 0] for tag in (0, 1, 2)}
        for (tag, _), (values, count) in self._losses.items():
            total, nonfinite = values.cpu().tolist()
            totals[tag][0] += total
            totals[tag][1] += count
            totals[tag][2] += int(nonfinite)

        def mean_loss(statistics):
            total, count, nonfinite = statistics
            return total / count if count and not nonfinite and math.isfinite(total) else None

        combined = [sum(values[position] for values in totals.values()) for position in range(3)]
        with self._lock:
            record = {"round": self._active_round, "train_loss": mean_loss(combined),
                      "train_steps": combined[1], "nonfinite_loss_steps": combined[2],
                      "participating_clients": len(self._clients),
                      "labeled_train_loss": mean_loss(totals[0]),
                      "unlabeled_train_loss": mean_loss(totals[2]), "mixed_train_loss": mean_loss(totals[1]),
                      "training_time_seconds": duration,
                      "cumulative_training_time_seconds": self.training_time_seconds,
                      "elapsed_time_seconds": time.perf_counter() - self._start_time,
                      "cpu_peak_rss_mib": self._round_peak_rss / MIB,
                      "cpu_tree_peak_rss_mib": self._round_peak_tree_rss / MIB,
                      "cuda_peak_allocated_mib": allocated, "cuda_peak_reserved_mib": reserved}
            self._active_round = None
        self._losses = {}
        self._clients = set()
        if self._active_warmup:
            record["phase"] = "warmup"
        if completed:
            (self.warmup if self._active_warmup else self.rounds).append(record)
        return record

    def snapshot(self):
        with self._lock:
            memory = {"cpu_peak_rss_mib": self._peak_rss / MIB,
                      "cpu_tree_peak_rss_mib": self._peak_tree_rss / MIB,
                      "cuda_peak_allocated_mib": self._cuda_allocated,
                      "cuda_peak_reserved_mib": self._cuda_reserved,
                      "rss_sample_interval_seconds": self.sample_interval, "rss_samples": self._samples}
            errors = sorted(self.measurement_errors)
        return {"format": "betelgeuse.training_metrics.v1", "method": self.args.method,
                "dataset": self.args.dataset, "save_name": self.args.save_name, "run": self.run_number,
                "seed": self.seed, "device": str(self.device), "status": self.status,
                "started_at": self.started_at, "finished_at": self.finished_at,
                "expected_rounds": self.args.num_global_iters, "completed_rounds": len(self.rounds),
                "training_time_seconds": self.training_time_seconds,
                "total_time_seconds": (self.total_time_seconds if self.total_time_seconds is not None else
                                       time.perf_counter() - self._start_time),
                "memory_footprint": memory, "measurement_errors": errors, "error_type": self.error_type,
                "definitions": {
                    "train_loss": "Equal-weight mean of actual unscaled minibatch objectives submitted to backward, including method coefficients; not validation loss or a full-dataset evaluation",
                    "train_steps": "Number of objective observations, not guaranteed optimizer steps if AMP skips an update",
                    "training_time_seconds": "Client selection, local training, aggregation and existing in-round diagnostics; includes any recorded pretraining warmup and interrupted round; excludes server initialization, evaluation and metrics serialization",
                    "total_time_seconds": "Wall time from before server construction through process completion, including evaluation and periodic output; excludes final recorder serialization and teardown",
                    "memory": "MiB (1024**2 bytes); sampled process RSS and sum of process-tree RSS (shared pages may be counted more than once); CUDA peaks cover this process's PyTorch allocator on the selected device, not all GPU usage",
                    "round_peaks": "Reset per training phase; run-level peaks also include initialization and evaluation",
                },
                "rounds": self.rounds, "warmup": self.warmup, "interrupted_round": self.interrupted_round}

    def save(self):
        with atomic_output(self.filename, "w") as stream:
            json.dump(self.snapshot(), stream, ensure_ascii=False, indent=2, allow_nan=False)
        with atomic_output(self.filename.with_suffix(".csv"), "w") as stream:
            writer = csv.DictWriter(stream, fieldnames=CURVE_FIELDS)
            writer.writeheader()
            writer.writerows(self.rounds)
        curves = {field: np.asarray([record[field] if record[field] is not None else np.nan
                                    for record in self.rounds]) for field in CURVE_FIELDS}
        with atomic_output(self.filename.with_suffix(".npz")) as stream:
            np.savez_compressed(stream, **curves)
