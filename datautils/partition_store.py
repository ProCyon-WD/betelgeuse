"""Shared, append-only image storage for trusted local experiment partitions."""

import contextlib
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import tempfile

import numpy as np


FORMAT = "betelgeuse.partition.indices.v1"
CHUNK_SIZE = 512


def partition_directory():
    default = Path(__file__).resolve().parents[1] / "data"
    return Path(os.environ.get("BETELGEUSE_DATA_DIR", default)).expanduser().resolve()


def partition_exists(filename):
    filename = Path(filename)
    return (filename.is_file() or _index_filename(filename).is_file()
            or Path(str(filename) + ".gz").is_file())


def _index_filename(filename):
    filename = Path(filename)
    if filename.suffix == ".gz":
        filename = filename.with_suffix("")
    return filename.with_suffix(".npz")


def read_partition_file(filename):
    filename = Path(filename)
    if not filename.exists() and _index_filename(filename).is_file():
        filename = _index_filename(filename)
    if not filename.exists() and Path(str(filename) + ".gz").is_file():
        filename = Path(str(filename) + ".gz")
    if filename.suffix == ".npz":
        with open(filename, "rb") as stream:
            return _read_index_archive(stream)
    opener = gzip.open if filename.suffix == ".gz" else open
    with opener(filename, "rb") as stream:
        payload = np.load(stream, allow_pickle=True).item()
    if not isinstance(payload, dict):
        raise ValueError("Expected a partition dictionary: {}".format(filename))
    return payload


def is_index_partition(payload):
    return payload.get("format") == FORMAT


def _write_index_archive(stream, payload):
    header = {key: value for key, value in payload.items() if key != "clients"}
    header["clients"] = []
    arrays = {}
    for position, (client_id, client) in enumerate(payload["clients"].items()):
        if isinstance(client_id, (int, np.integer)):
            client_id = int(client_id)
        elif not isinstance(client_id, str):
            raise ValueError("Client IDs must be integers or strings")
        prefix = "client_{}".format(position)
        header["clients"].append({"id": client_id, "prefix": prefix,
                                  "image_sha256": client["image_sha256"],
                                  "target_sha256": client["target_sha256"]})
        arrays[prefix + "_indices"] = client["indices"]
        arrays[prefix + "_target"] = client["target"]
    arrays["metadata"] = np.frombuffer(json.dumps(header).encode("utf-8"), dtype=np.uint8)
    np.savez_compressed(stream, **arrays)


def _read_index_archive(stream):
    with np.load(stream, allow_pickle=False) as archive:
        payload = json.loads(archive["metadata"].tobytes().decode("utf-8"))
        if not is_index_partition(payload):
            raise ValueError("Unknown partition archive format")
        entries = payload.pop("clients")
        payload["clients"] = {}
        for entry in entries:
            prefix = entry["prefix"]
            if entry["id"] in payload["clients"]:
                raise ValueError("Duplicate client ID in partition archive")
            payload["clients"][entry["id"]] = {
                "indices": archive[prefix + "_indices"], "target": archive[prefix + "_target"],
                "image_sha256": entry["image_sha256"], "target_sha256": entry["target_sha256"],
            }
    return payload


@contextlib.contextmanager
def atomic_output(filename, mode="wb"):
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix="." + filename.name + ".", dir=filename.parent)
    try:
        with os.fdopen(descriptor, mode) as stream:
            yield stream
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, filename)
        directory_fd = os.open(str(filename.parent), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _schema(data_map):
    schema = None
    if not data_map:
        raise ValueError("Cannot store an empty partition")
    for client_id, client in data_map.items():
        if set(client) != {"data", "target"}:
            raise ValueError("Unsupported fields for client {}".format(client_id))
        images, targets = client["data"], client["target"]
        if not isinstance(images, np.ndarray) or images.ndim < 2 or images.dtype.hasobject:
            raise ValueError("Expected numeric image arrays for client {}".format(client_id))
        if not isinstance(targets, np.ndarray) or targets.ndim < 1 or targets.dtype.hasobject:
            raise ValueError("Expected numeric target arrays for client {}".format(client_id))
        if len(images) != len(targets):
            raise ValueError("Image/target length mismatch for client {}".format(client_id))
        current = {"version": 1, "dtype": images.dtype.str, "image_shape": list(images.shape[1:])}
        if schema is not None and current != schema:
            raise ValueError("All clients in a dataset must use the same image shape and dtype")
        schema = current
    if not all(size > 0 for size in schema["image_shape"]):
        raise ValueError("Image dimensions must be positive")
    return schema


def _array_digest(array):
    digest = hashlib.sha256()
    for start in range(0, len(array), CHUNK_SIZE):
        digest.update(np.ascontiguousarray(array[start:start + CHUNK_SIZE]).tobytes())
    return digest.hexdigest()


class IndexedImages:
    """Array-like image lookup without copying the shared bank into each client."""

    def __init__(self, bank_path, image_shape, dtype, indices):
        self.bank_path = str(Path(bank_path).resolve())
        self.image_shape = tuple(image_shape)
        self.dtype = np.dtype(dtype)
        self.indices = np.array(indices, dtype=np.int64, copy=True)
        self._images = None

    @property
    def shape(self):
        return (len(self.indices),) + self.image_shape

    @property
    def ndim(self):
        return len(self.shape)

    def __len__(self):
        return len(self.indices)

    def _bank(self):
        if self._images is None:
            row_bytes = int(np.prod(self.image_shape)) * self.dtype.itemsize
            count = os.path.getsize(self.bank_path) // row_bytes
            if count:
                self._images = np.memmap(self.bank_path, dtype=self.dtype, mode="r",
                                         shape=(count,) + self.image_shape)
            else:
                self._images = np.empty((0,) + self.image_shape, dtype=self.dtype)
        return self._images

    def __getitem__(self, index):
        return np.array(self._bank()[self.indices[index]], copy=True)

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("Indexed images require a copy to materialize")
        return np.asarray(self[:], dtype=dtype)

    def __deepcopy__(self, memo):
        result = type(self)(self.bank_path, self.image_shape, self.dtype, self.indices)
        result._images = self._images
        if memo is not None:
            memo[id(self)] = result
        return result

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_images"] = None
        return state


def _load_index_payload(filename, payload):
    parent = Path(filename).resolve().parent
    store = (parent / payload["store"]).resolve()
    if parent not in store.parents:
        raise ValueError("Image store must be inside the partition directory")
    with open(store / "metadata.json", encoding="utf-8") as stream:
        metadata = json.load(stream)
    if metadata != payload["image_schema"]:
        raise ValueError("Image store schema does not match partition metadata")
    row_bytes = int(np.prod(metadata["image_shape"])) * np.dtype(metadata["dtype"]).itemsize
    bank_path = store / "images.bin"
    count = bank_path.stat().st_size // row_bytes
    result = {}
    for client_id, client in payload["clients"].items():
        indices = client["indices"]
        targets = client["target"]
        if (not isinstance(indices, np.ndarray) or indices.ndim != 1
                or indices.dtype.kind not in "iu" or len(indices) != len(targets)):
            raise ValueError("Invalid indices for client {}".format(client_id))
        if len(indices) and (indices.min() < 0 or indices.max() >= count):
            raise ValueError("Image bank is missing rows for client {}".format(client_id))
        result[client_id] = {
            "data": IndexedImages(bank_path, metadata["image_shape"], metadata["dtype"], indices),
            "target": targets,
        }
    return result


def load_partition(filename):
    payload = read_partition_file(filename)
    if is_index_partition(payload):
        return _load_index_payload(filename, payload)
    return payload


def _verify_payload(filename, payload, original=None):
    restored = _load_index_payload(filename, payload)
    if original is not None and list(restored) != list(original):
        raise ValueError("Client IDs or client order changed")
    for client_id, client in restored.items():
        recorded = payload["clients"][client_id]
        images, targets = client["data"], client["target"]
        if _array_digest(images) != recorded["image_sha256"]:
            raise ValueError("Image checksum mismatch for client {}".format(client_id))
        if _array_digest(targets) != recorded["target_sha256"]:
            raise ValueError("Target checksum mismatch for client {}".format(client_id))
        if original is not None:
            expected = original[client_id]
            for field, actual in (("data", images), ("target", targets)):
                reference = expected[field]
                if actual.shape != reference.shape or actual.dtype != reference.dtype:
                    raise ValueError("Shape/dtype changed for client {}: {}".format(client_id, field))
                for start in range(0, len(reference), CHUNK_SIZE):
                    stop = start + CHUNK_SIZE
                    if actual[start:stop].tobytes() != reference[start:stop].tobytes():
                        raise ValueError("Data changed for client {}: {}".format(client_id, field))
    return {"clients": len(restored), "samples": sum(len(client["data"]) for client in restored.values())}


def verify_partition(filename, original=None):
    payload = read_partition_file(filename)
    if not is_index_partition(payload):
        raise ValueError("Not an indexed partition: {}".format(filename))
    return _verify_payload(filename, payload, original)


def save_partition(filename, data_map):
    filename = Path(filename).resolve()
    if filename.suffix not in (".npy", ".npz"):
        raise ValueError("Partition filename must end in .npy or .npz")
    filename = _index_filename(filename)
    dataset = filename.name.split("_", 1)[0]
    if not re.fullmatch(r"[A-Za-z0-9-]+", dataset):
        raise ValueError("Invalid dataset name")
    schema = _schema(data_map)
    store = filename.parent / "_shared" / dataset
    store.mkdir(parents=True, exist_ok=True)
    with open(store / ".lock", "a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _save_locked(filename, data_map, store, schema)


def _save_locked(filename, data_map, store, schema):
    metadata_path = store / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as stream:
            if json.load(stream) != schema:
                raise ValueError("Dataset image shape/dtype differs from the existing bank")
    else:
        with atomic_output(metadata_path, "w") as stream:
            json.dump(schema, stream, sort_keys=True)
    row_bytes = int(np.prod(schema["image_shape"])) * np.dtype(schema["dtype"]).itemsize
    connection = sqlite3.connect(str(store / "index.sqlite3"))
    try:
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("CREATE TABLE IF NOT EXISTS images (digest BLOB PRIMARY KEY, sample_index INTEGER UNIQUE NOT NULL)")
        connection.commit()
        lookup = dict(connection.execute("SELECT digest, sample_index FROM images"))
        count = len(lookup)
        if count and (min(lookup.values()) != 0 or max(lookup.values()) != count - 1):
            raise ValueError("Non-contiguous shared image index")
        bank_path = store / "images.bin"
        bank_path.touch(exist_ok=True)
        committed_bytes = count * row_bytes
        if bank_path.stat().st_size < committed_bytes:
            raise ValueError("Shared image bank is truncated")
        payload = {"format": FORMAT, "store": store.relative_to(filename.parent).as_posix(),
                   "image_schema": schema, "clients": {}}
        with open(bank_path, "r+b") as bank:
            bank.truncate(committed_bytes)
            bank.seek(committed_bytes)
            try:
                connection.execute("BEGIN IMMEDIATE")
                for client_id, client in data_map.items():
                    images, targets = client["data"], client["target"]
                    indices = np.empty(len(images), dtype=np.int64)
                    image_digest = hashlib.sha256()
                    for position, image in enumerate(images):
                        image_bytes = image.tobytes(order="C")
                        image_digest.update(image_bytes)
                        digest = hashlib.sha256(image_bytes).digest()
                        sample_index = lookup.get(digest)
                        if sample_index is None:
                            sample_index = len(lookup)
                            bank.write(image_bytes)
                            connection.execute("INSERT INTO images VALUES (?, ?)", (digest, sample_index))
                            lookup[digest] = sample_index
                        indices[position] = sample_index
                    payload["clients"][client_id] = {
                        "indices": indices, "target": targets.copy(),
                        "image_sha256": image_digest.hexdigest(),
                        "target_sha256": _array_digest(targets),
                    }
                bank.flush()
                os.fsync(bank.fileno())
                connection.commit()
            except BaseException:
                connection.rollback()
                durable_count = connection.execute("SELECT COUNT(*) FROM images").fetchone()[0]
                bank.truncate(durable_count * row_bytes)
                bank.flush()
                os.fsync(bank.fileno())
                raise
        with atomic_output(filename, "w+b") as stream:
            _write_index_archive(stream, payload)
            stream.flush()
            stream.seek(0)
            serialized = _read_index_archive(stream)
            _verify_payload(filename, serialized, original=data_map)
        return {"clients": len(data_map), "samples": sum(len(client["data"]) for client in data_map.values()),
                "new_images": len(lookup) - count, "unique_images": len(lookup),
                "image_bytes_added": (len(lookup) - count) * row_bytes,
                "index_bytes": filename.stat().st_size, "verified": True}
    finally:
        connection.close()
