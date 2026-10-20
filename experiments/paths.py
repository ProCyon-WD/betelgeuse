"""Project-relative paths shared by launchers and server implementations."""

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def source_directory(dataset, root=None):
    root = root or os.environ.get("BETELGEUSE_SOURCE_DIR", "data/datasource")
    return project_path(root) / dataset.lower()


def metrics_directory(args):
    path = project_path(getattr(args, "metrics_dir", "outputs/metrics"))
    path.mkdir(parents=True, exist_ok=True)
    return str(path) + os.sep


def log_directory():
    path = PROJECT_ROOT / "outputs" / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def feature_directory():
    return project_path(os.environ.get("BETELGEUSE_CLUSTER_DIR", "config/feature_partitions"))
