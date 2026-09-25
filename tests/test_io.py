import pytest

from src.utils.io import check_not_lfs_pointer, is_lfs_pointer, load_json, save_json

POINTER_TEXT = (
    "version https://git-lfs.github.com/spec/v1\n"
    "oid sha256:e9e18de2740c8a4d1a6a3b3c6f5c8e2d1a0b9c8d7e6f5a4b3c2d1e0f9a8b7c6d\n"
    "size 614836\n"
)


def test_pointer_file_detected(tmp_path):
    p = tmp_path / "pipeline.joblib"
    p.write_text(POINTER_TEXT, encoding="utf-8")
    assert is_lfs_pointer(p) is True
    with pytest.raises(RuntimeError, match="git lfs pull"):
        check_not_lfs_pointer(p, "LR pipeline")


def test_normal_small_file_not_pointer(tmp_path):
    p = tmp_path / "small.bin"
    p.write_bytes(b"\x80\x04\x95 not a pointer")
    assert is_lfs_pointer(p) is False
    check_not_lfs_pointer(p)


def test_large_file_not_pointer_even_with_prefix(tmp_path):
    p = tmp_path / "big.bin"
    p.write_bytes(b"version https://git-lfs" + b"x" * 2048)
    assert is_lfs_pointer(p) is False


def test_missing_file_not_pointer(tmp_path):
    assert is_lfs_pointer(tmp_path / "nope.bin") is False


def test_save_and_load_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "out.json"
    save_json({"a": 1, "b": [1, 2]}, path)
    assert load_json(path) == {"a": 1, "b": [1, 2]}
