import json
from pathlib import Path

LFS_POINTER_PREFIX = b"version https://git-lfs"
LFS_POINTER_MAX_BYTES = 1024


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def is_lfs_pointer(path) -> bool:
    """Return True if path is an unresolved Git LFS pointer file.

    A pointer is a small text file (about 130 bytes) starting with
    "version https://git-lfs.github.com/spec/v1". Real model weights are
    at least hundreds of KB, so anything under 1 KB with that prefix is a
    pointer left behind by a clone without `git lfs pull`.
    """
    p = Path(path)
    if not p.is_file():
        return False
    if p.stat().st_size >= LFS_POINTER_MAX_BYTES:
        return False
    with open(p, "rb") as f:
        head = f.read(len(LFS_POINTER_PREFIX))
    return head == LFS_POINTER_PREFIX


def check_not_lfs_pointer(path, what="model file"):
    """Raise a clear error if path is an LFS pointer instead of the real file."""
    if is_lfs_pointer(path):
        raise RuntimeError(
            f"{what} at '{path}' is a Git LFS pointer, not the real file. "
            "Install Git LFS and run `git lfs pull` from the repository root."
        )
