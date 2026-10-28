"""Tests for scripts/make_results_table.py, which builds the README results table."""
import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent

# scripts/ is not a package, so load the module from its file path.
_spec = importlib.util.spec_from_file_location("make_results_table", REPO / "scripts" / "make_results_table.py")
make_results_table = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(make_results_table)
build_table = make_results_table.build_table

HEADER = "| Model | Accuracy | Macro F1 | Macro Recall |\n|---|---|---|---|"


def _write(results_dir, name, test=None, validation=None, **extra):
    data = dict(extra)
    if test is not None:
        data["test"] = test
    if validation is not None:
        data["validation"] = validation
    (results_dir / f"{name}.json").write_text(json.dumps(data), encoding="utf-8")


def _metrics(acc, f1, recall):
    return {"accuracy": acc, "f1_macro": f1, "recall_macro": recall}


def _rows(table):
    return table.splitlines()[2:]


def test_rows_follow_order(tmp_path):
    # Written in reverse so file order cannot explain the result.
    for name in reversed(make_results_table.ORDER):
        _write(tmp_path, name, test=_metrics(0.5, 0.5, 0.5))
    names = [row.split(" | ")[0].lstrip("| ") for row in _rows(build_table(tmp_path))]
    assert names == [make_results_table.DISPLAY_NAMES[k] for k in make_results_table.ORDER]


def test_unknown_keys_appended_after_known(tmp_path):
    _write(tmp_path, "zeta", test=_metrics(0.1, 0.1, 0.1))
    _write(tmp_path, "alpha", test=_metrics(0.2, 0.2, 0.2))
    _write(tmp_path, "lr", test=_metrics(0.3, 0.3, 0.3))
    rows = _rows(build_table(tmp_path))
    assert rows[0].startswith(f"| {make_results_table.DISPLAY_NAMES['lr']} |")
    assert rows[1].startswith("| alpha |")
    assert rows[2].startswith("| zeta |")


def test_files_without_split_are_skipped(tmp_path):
    _write(tmp_path, "lr", test=_metrics(0.5922, 0.5887, 0.5987))
    # Shaped like results/roberta_trainer_state_*.json: no validation or test block.
    _write(tmp_path, "roberta_trainer_state_2025-12-05", best_metric=0.78, log_history=[])
    _write(tmp_path, "gru", validation=_metrics(0.6, 0.6, 0.6))
    table = build_table(tmp_path, split="test")
    assert _rows(table) == ["| TF-IDF + Logistic Regression | 0.5922 | 0.5887 | 0.5987 |"]


def test_validation_split(tmp_path):
    _write(tmp_path, "gru", validation=_metrics(0.61451, 0.60361, 0.65561))
    assert _rows(build_table(tmp_path, split="validation")) == ["| BiGRU | 0.6145 | 0.6036 | 0.6556 |"]


def test_missing_metrics_render_as_dash(tmp_path):
    _write(tmp_path, "lstm", test={"accuracy": 0.57851})
    assert _rows(build_table(tmp_path)) == ["| BiLSTM | 0.5785 | - | - |"]


@pytest.mark.parametrize("make_file", [False, True])
def test_empty_directory_gives_placeholder(tmp_path, make_file):
    if make_file:
        _write(tmp_path, "trainer_state", log_history=[])
    assert build_table(tmp_path) == HEADER + "\n| run training to populate | - | - | - |"
