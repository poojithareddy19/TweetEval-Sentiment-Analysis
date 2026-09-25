import warnings

import pytest

from src.utils.metrics import compute_metrics

EXPECTED_KEYS = {"accuracy", "f1_macro", "recall_macro", "f1_per_class", "confusion_matrix", "classification_report"}


def test_all_keys_present_and_perfect_scores():
    y = [0, 1, 2, 0, 1, 2]
    m = compute_metrics(y, y)
    assert set(m) == EXPECTED_KEYS
    assert m["accuracy"] == 1.0
    assert m["f1_macro"] == 1.0
    assert m["recall_macro"] == 1.0
    assert m["f1_per_class"] == [1.0, 1.0, 1.0]
    assert m["confusion_matrix"] == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert set(m["classification_report"]) >= {"negative", "neutral", "positive", "macro avg"}


def test_recall_macro_is_mean_of_per_class_recall():
    y_true = [0, 0, 1, 1, 2, 2]
    y_pred = [0, 1, 1, 1, 2, 0]
    m = compute_metrics(y_true, y_pred)
    # recall: negative 1/2, neutral 2/2, positive 1/2
    assert m["recall_macro"] == pytest.approx((0.5 + 1.0 + 0.5) / 3)


def test_missing_class_in_predictions_is_handled():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 0, 0, 1, 1]  # class 2 never predicted
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = compute_metrics(y_true, y_pred)
    assert set(m) == EXPECTED_KEYS
    assert len(m["f1_per_class"]) == 3
    assert m["f1_per_class"][2] == 0.0
    assert len(m["confusion_matrix"]) == 3
    assert m["classification_report"]["positive"]["recall"] == 0.0
    assert m["recall_macro"] == pytest.approx(2 / 3)


def test_missing_class_in_truth_is_handled():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 2, 1, 1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = compute_metrics(y_true, y_pred)
    assert len(m["confusion_matrix"]) == 3
    assert "positive" in m["classification_report"]
