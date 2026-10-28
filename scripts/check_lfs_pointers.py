"""Exit with an error if any file under models/ is an unresolved Git LFS pointer.

Usage (from the repo root):
    python scripts/check_lfs_pointers.py [models_dir]
"""
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.utils.io import is_lfs_pointer  # noqa: E402


def find_pointers(models_dir):
    return [p for p in sorted(Path(models_dir).rglob("*")) if is_lfs_pointer(p)]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    models_dir = Path(argv[0]) if argv else REPO / "models"
    pointers = find_pointers(models_dir)
    if pointers:
        print("Unresolved Git LFS pointers (run `git lfs pull` before building or testing):")
        for p in pointers:
            print(f"  {p}")
        return 1
    print(f"No LFS pointers under {models_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
