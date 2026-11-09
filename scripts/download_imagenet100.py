"""Download the pinned ImageNet-100 community mirror without GitHub or CUDA."""

import argparse
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import sys
import tempfile
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen
import zipfile
import zlib


PROJECT = Path(__file__).resolve().parents[1]
SPECIFICATION = PROJECT / "config/datasets/imagenet100_download.json"
CHUNK_SIZE = 1024 * 1024
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp"}


def sha256_file(filename):
    digest = hashlib.sha256()
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def file_matches(filename, specification):
    return (filename.is_file() and filename.stat().st_size == specification["size"]
            and sha256_file(filename) == specification["sha256"])


@contextmanager
def directory_lock(directory):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".download.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another downloader is using {directory}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def download_file(url, destination, specification, retries=5, timeout=60):
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.is_symlink() or partial.is_symlink():
        raise ValueError(f"Refusing a symlink in the download cache: {destination}")
    if destination.exists():
        if not file_matches(destination, specification):
            raise ValueError(f"Cached file failed SHA256 validation: {destination}; move it aside and retry")
        print(f"REUSE {destination.name} (SHA256 verified)", flush=True)
        return destination
    size = specification["size"]
    for attempt in range(retries + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            raise ValueError(f"Oversized partial download: {partial}; move it aside and retry")
        if offset == size:
            if not file_matches(partial, specification):
                raise ValueError(f"Downloaded SHA256 mismatch: {partial}; remove this .part file and retry")
            partial.replace(destination)
            print(f"VERIFIED {destination.name}", flush=True)
            return destination
        headers = {"User-Agent": "betelgeuse-dataset-downloader", "Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        try:
            with urlopen(Request(url, headers=headers), timeout=timeout) as response:
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected HTTP content encoding; cannot safely resume")
                if response.status == 206:
                    match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                    if (match is None or int(match[1]) != offset or int(match[3]) != size
                            or not offset <= int(match[2]) < size):
                        raise ValueError("Server returned an invalid Content-Range; refusing to append")
                    end = int(match[2]) + 1
                elif response.status == 200:
                    if offset:
                        print(f"Server ignored Range; restarting {destination.name}", flush=True)
                    offset = 0
                    end = size
                else:
                    raise ValueError(f"Unexpected HTTP status {response.status}")
                content_length = response.headers.get("Content-Length")
                if content_length is not None and int(content_length) != end - offset:
                    raise ValueError("Unexpected HTTP Content-Length; refusing this response")
                last_report = time.monotonic()
                print(f"DOWNLOAD {destination.name}: {offset / size:.1%}", flush=True)
                with partial.open("ab" if offset else "wb") as stream:
                    while block := response.read(CHUNK_SIZE):
                        if offset + len(block) > end:
                            raise ValueError("Response exceeds the expected archive size")
                        stream.write(block)
                        offset += len(block)
                        if time.monotonic() - last_report >= 5:
                            print(f"  {destination.name}: {offset / size:.1%}", flush=True)
                            last_report = time.monotonic()
                if offset != size:
                    raise ConnectionError(f"Incomplete response: {offset}/{size} bytes")
            if not file_matches(partial, specification):
                raise ValueError(f"Downloaded SHA256 mismatch: {partial}; remove this .part file and retry")
            partial.replace(destination)
            print(f"VERIFIED {destination.name}", flush=True)
            return destination
        except HTTPError as error:
            if error.code in (401, 403, 404):
                raise RuntimeError(f"Source returned HTTP {error.code} for {destination.name}. "
                                   "Check source availability/access; no access controls are bypassed.") from error
            if error.code not in (408, 429, 500, 502, 503, 504) or attempt == retries:
                raise
            print(f"HTTP {error.code}; retry {attempt + 1}/{retries}", flush=True)
        except (URLError, TimeoutError, ConnectionError, http.client.IncompleteRead) as error:
            if attempt == retries:
                raise RuntimeError(f"Download failed for {destination.name}: {error}. "
                                   "Partial bytes are retained. Check access to the source website/proxy, "
                                   "then rerun the same command.") from error
            print(f"Network interruption; retry {attempt + 1}/{retries}: {error}", flush=True)
        time.sleep(min(2 ** attempt, 15))


def safe_member_path(member, classes, output):
    path = PurePosixPath(member.filename)
    if path.is_absolute() or ".." in path.parts or "\\" in member.filename or not path.parts:
        raise ValueError(f"Unsafe archive path: {member.filename}")
    mode = member.external_attr >> 16
    if stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR):
        raise ValueError(f"Archive contains a symlink or special file: {member.filename}")
    if path.parts[0] not in ("train", "val"):
        raise ValueError(f"Unexpected archive split: {member.filename}")
    if len(path.parts) > 1 and path.parts[1] not in classes:
        raise ValueError(f"Unexpected archive class: {member.filename}")
    if not member.is_dir() and (len(path.parts) != 3 or path.suffix.lower() not in IMAGE_EXTENSIONS):
        raise ValueError(f"Unexpected non-image archive member: {member.filename}")
    destination = output.joinpath(*path.parts)
    for parent in (destination, *destination.parents):
        if parent == output:
            break
        if parent.is_symlink():
            raise ValueError(f"Refusing existing output symlink: {parent}")
    return path, destination


def matches_zip_member(filename, member):
    if not filename.is_file() or filename.stat().st_size != member.file_size:
        return False
    checksum = 0
    with filename.open("rb") as stream:
        for block in iter(lambda: stream.read(CHUNK_SIZE), b""):
            checksum = zlib.crc32(block, checksum)
    return checksum == member.CRC


def extract_archive(filename, output, classes, seen):
    extracted = 0
    reused = 0
    with zipfile.ZipFile(filename) as archive:
        entries = []
        current_names = set()
        for member in archive.infolist():
            relative, destination = safe_member_path(member, classes, output)
            if member.is_dir():
                continue
            name = relative.as_posix()
            if name in seen or name in current_names:
                raise ValueError(f"Duplicate image across archives: {name}")
            current_names.add(name)
            entries.append((member, destination))
        for member, destination in entries:
            if destination.exists():
                if not matches_zip_member(destination, member):
                    raise ValueError(f"Existing image differs from pinned source: {destination}; "
                                     "use a separate output directory or move the conflicting file aside")
                reused += 1
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary = tempfile.mkstemp(prefix=".download-", suffix=".part", dir=destination.parent)
            temporary = Path(temporary)
            try:
                with os.fdopen(descriptor, "wb") as target, archive.open(member) as source:
                    shutil.copyfileobj(source, target, CHUNK_SIZE)
                if temporary.stat().st_size != member.file_size:
                    raise ValueError(f"Extracted image size mismatch: {member.filename}")
                temporary.replace(destination)
            finally:
                temporary.unlink(missing_ok=True)
            extracted += 1
        seen.update(current_names)
    print(f"EXTRACT {filename.name}: {extracted} new, {reused} verified existing images", flush=True)


def verify_dataset(output, specification, expected_paths=None, decode=False):
    if decode:
        from PIL import Image
    totals = {}
    actual_paths = set()
    for split, expected_counts in specification["counts"].items():
        counts = Counter()
        root = output / split
        if not root.is_dir() or root.is_symlink():
            raise ValueError(f"Missing or symlinked split: {root}")
        for directory, folders, files in os.walk(root, followlinks=False):
            for name in folders + files:
                if (Path(directory) / name).is_symlink():
                    raise ValueError(f"Unexpected output symlink: {Path(directory) / name}")
            for name in files:
                filename = Path(directory) / name
                if filename.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                relative = filename.relative_to(output)
                if len(relative.parts) != 3 or relative.parts[1] not in expected_counts:
                    raise ValueError(f"Unexpected image: {filename}")
                counts[relative.parts[1]] += 1
                actual_paths.add(relative.as_posix())
                if decode:
                    with Image.open(filename) as image:
                        image.convert("RGB").load()
        if dict(counts) != expected_counts:
            differences = {name: {"expected": count, "actual": counts[name]}
                           for name, count in expected_counts.items() if counts[name] != count}
            raise ValueError(f"{split} class counts differ from the pinned dataset: {differences}")
        totals[split] = sum(counts.values())
    if expected_paths is not None and actual_paths != expected_paths:
        raise ValueError("Output image paths do not exactly match the pinned archives")
    return totals


def write_json(filename, value):
    descriptor, temporary = tempfile.mkstemp(prefix=".download-", dir=filename.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
        temporary.replace(filename)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=PROJECT / "data/datasource/imagenet100")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT / "data/downloads/imagenet100")
    parser.add_argument("--endpoint", default="https://huggingface.co", help="HTTPS dataset host; pinned checksums remain mandatory")
    parser.add_argument("--timeout", type=int, default=60, help="Network socket timeout in seconds")
    parser.add_argument("--retries", type=int, default=5, help="Retries per archive after the initial attempt")
    parser.add_argument("--dry-run", action="store_true", help="Show paths, source and size without downloading")
    parser.add_argument("--verify-only", action="store_true", help="Offline image count/layout check, not an archive hash check")
    parser.add_argument("--verify-images", action="store_true", help="Also fully RGB-decode every image with Pillow")
    args = parser.parse_args(argv)
    if args.timeout < 1 or args.retries < 0:
        parser.error("timeout must be positive and retries nonnegative")
    endpoint = args.endpoint.rstrip("/")
    parsed_endpoint = urlparse(endpoint)
    if (parsed_endpoint.scheme != "https" or not parsed_endpoint.netloc or parsed_endpoint.username
            or parsed_endpoint.password or parsed_endpoint.path or parsed_endpoint.query or parsed_endpoint.fragment):
        parser.error("endpoint must be an HTTPS origin without credentials, path, query or fragment")
    output = args.output.expanduser().resolve()
    cache = args.cache_dir.expanduser().resolve()
    if output == cache or output.is_relative_to(cache) or cache.is_relative_to(output):
        parser.error("output and cache-dir must be separate, non-nested directories")
    specification = json.loads(SPECIFICATION.read_text(encoding="utf-8"))
    classes = (PROJECT / "config/datasets/imagenet100_cmc.txt").read_text().split()
    if classes != specification["classes"]:
        raise ValueError("Project class order differs from the pinned download specification")
    compressed_size = sum(item["size"] for item in specification["archives"])
    print(f"SOURCE {endpoint}/datasets/{specification['repository']} @ {specification['revision']}", flush=True)
    print("Community mirror, NOT an official ImageNet download. Follow upstream data-use terms.", flush=True)
    print(f"OUTPUT {output}\nCACHE {cache}\nDOWNLOAD {compressed_size / 1024 ** 3:.2f} GiB; "
          "allow at least 20 GiB for archives plus extracted images.", flush=True)
    if args.dry_run:
        return
    if args.verify_images:
        import PIL.Image
    if args.verify_only:
        print("Verified counts/layout" + (" and RGB decoding" if args.verify_images else "") + ": "
              + json.dumps(verify_dataset(output, specification, decode=args.verify_images)))
        return
    with directory_lock(cache), directory_lock(output):
        seen = set()
        for item in specification["archives"]:
            url = f"{endpoint}/datasets/{specification['repository']}/resolve/{specification['revision']}/{item['filename']}"
            filename = download_file(url, cache / item["filename"], item, args.retries, args.timeout)
            extract_archive(filename, output, classes, seen)
        totals = verify_dataset(output, specification, seen, args.verify_images)
        provenance = dict(specification, endpoint=endpoint, completed_at=datetime.now(timezone.utc).isoformat(),
                          train_images=totals["train"], val_images=totals["val"],
                          validation=["archive sizes and SHA256", "image CRC32", "exact image paths and per-class counts"],
                          all_images_rgb_decoded=args.verify_images)
        write_json(output / "download_manifest.json", provenance)
        if not (output / "source.json").exists():
            write_json(output / "source.json", provenance)
        print(f"READY {totals['train']} train + {totals['val']} val images in {output}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Verified files and partial downloads are retained; rerun the same command.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError, zipfile.BadZipFile, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        sys.exit(1)
