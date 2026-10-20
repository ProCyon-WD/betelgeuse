"""Migrate, verify, or export trusted local federated dataset caches."""

import argparse
from datetime import datetime, timezone
import gc
import json
from pathlib import Path

import numpy as np

from datautils.partition_store import (
    atomic_output,
    is_index_partition,
    load_partition,
    partition_directory,
    read_partition_file,
    save_partition,
    verify_partition,
)


def migrate_file(source):
    source = Path(source).resolve()
    if not source.exists():
        candidate = source.with_suffix("") if source.suffix == ".gz" else source
        if candidate.with_suffix(".npz").is_file():
            source = candidate.with_suffix(".npz")
    original_bytes = source.stat().st_size
    original = read_partition_file(source)
    if is_index_partition(original):
        result = verify_partition(source)
        return dict(result, source=source.name, status="already_indexed", verified=True)
    destination = source.with_suffix("") if source.suffix == ".gz" else source
    destination = destination.with_suffix(".npz")
    if destination.exists():
        verify_partition(destination, original=original)
        result = {"clients": len(original), "samples": sum(len(client["data"]) for client in original.values()),
                  "index_bytes": destination.stat().st_size, "verified": True}
    else:
        result = save_partition(destination, original)
    verify_partition(destination, original=original)
    source.unlink()
    return dict(result, source=source.name, destination=destination.name,
                original_bytes=original_bytes, status="migrated",
                verified_at=datetime.now(timezone.utc).isoformat())


def export_partition(source, destination):
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("Refusing to overwrite export destination: {}".format(destination))
    partition = load_partition(source)
    materialized = {client_id: {"data": np.asarray(client["data"]), "target": client["target"]}
                    for client_id, client in partition.items()}
    with atomic_output(destination) as stream:
        np.save(stream, materialized, allow_pickle=True)


def discover_partitions(directory, names=None):
    if names:
        paths = [directory / name for name in names]
    else:
        paths = sorted(list(directory.glob("*.npy")) + list(directory.glob("*.npy.gz"))
                       + list(directory.glob("*.npz")))
    if not paths:
        raise ValueError("No partition files found in {}".format(directory))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("migrate", "verify", "export"))
    parser.add_argument("--data-dir", type=Path, default=partition_directory())
    parser.add_argument("--partition", action="append", help="A partition filename; repeat to select several")
    parser.add_argument("--output", type=Path, help="Destination for exporting a single legacy .npy")
    args = parser.parse_args()
    directory = args.data_dir.expanduser().resolve()
    paths = discover_partitions(directory, args.partition)
    if args.action == "export":
        if len(paths) != 1 or args.output is None:
            parser.error("export requires exactly one --partition and an --output path")
        export_partition(paths[0], args.output)
        print("Exported {}".format(args.output), flush=True)
        return
    if args.output is not None:
        parser.error("--output is only valid for export")
    report_path = directory / "migration_report.json"
    report = {"version": 1, "files": {}}
    if args.action == "migrate" and report_path.exists():
        with open(report_path, encoding="utf-8") as stream:
            report = json.load(stream)
    if args.action == "migrate" and "initial_partition_bytes" not in report:
        report["initial_partition_bytes"] = sum(source.stat().st_size for source in paths)
        report["started_at"] = datetime.now(timezone.utc).isoformat()
    for position, source in enumerate(paths, start=1):
        print("[{}/{}] {} {}".format(position, len(paths), args.action, source.name), flush=True)
        if args.action == "migrate":
            result = migrate_file(source)
            record_key = result.get("destination", source.name)
            if result["status"] == "migrated" or record_key not in report["files"]:
                report["files"][record_key] = result
            report["updated_at"] = datetime.now(timezone.utc).isoformat()
            with atomic_output(report_path, "w") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
        else:
            result = verify_partition(source)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)
        gc.collect()
    print("Completed: {} partitions".format(len(paths)), flush=True)


if __name__ == "__main__":
    main()
