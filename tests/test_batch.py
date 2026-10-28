"""Tests for src/models/batch.py (the app's batch CSV mode without Streamlit)."""
import math
from pathlib import Path

import numpy as np
import pytest

from src.models.batch import predict_proba_batch, prediction_columns
from src.models.inference import load_inference_config
from src.utils.io import is_lfs_pointer
from src.utils.preprocessing import preprocess_tweet

REPO = Path(__file__).resolve().parent.parent
LR_DIR = REPO / "models" / "lr"
LABELS = ["negative", "neutral", "positive"]


class _RecordingModel:
    """Stands in for a scikit-learn pipeline and records what it was given."""

    def __init__(self):
        self.seen = None

    def predict_proba(self, texts):
        self.seen = list(texts)
        return np.tile([0.2, 0.5, 0.3], (len(texts), 1))


@pytest.fixture(scope="module")
def lr_model():
    path = LR_DIR / "pipeline.joblib"
    if not path.exists() or is_lfs_pointer(path):
        pytest.skip("models/lr/pipeline.joblib missing or a Git LFS pointer")
    pytest.importorskip("sklearn")
    from src.models.inference import load_lr_pipeline

    return load_lr_pipeline(LR_DIR), load_inference_config(LR_DIR)


def test_lr_batch_matches_single_predictions(lr_model):
    pipe, cfg = lr_model
    texts = ["I love this so much :)", "worst day ever :(", "The meeting is at 10:30 tomorrow."]
    probs = predict_proba_batch("lr", pipe, texts, cfg)
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)
    for text, row in zip(texts, probs):
        single = pipe.predict_proba([preprocess_tweet(text)])[0]
        np.testing.assert_allclose(row, single, atol=1e-12)


def test_lr_batch_handles_nan(lr_model):
    pipe, cfg = lr_model
    probs = predict_proba_batch("lr", pipe, ["good", math.nan, None], cfg)
    assert probs.shape == (3, 3)
    np.testing.assert_allclose(probs[1], probs[2])


@pytest.mark.parametrize("kind", ["lr", "keras", "roberta"])
def test_empty_list(kind):
    probs = predict_proba_batch(kind, None, [], {"labels": LABELS, "preprocessing": "tweet"})
    assert probs.shape == (0, 3)


def test_non_string_cells_become_empty_text():
    model = _RecordingModel()
    cfg = {"labels": LABELS, "preprocessing": "tweet", "map_emoticons": False}
    predict_proba_batch("lr", model, ["Hello @bob", math.nan, None, 3.5], cfg)
    assert model.seen == ["hello <user>", "", "", ""]


def test_unknown_kind():
    with pytest.raises(ValueError):
        predict_proba_batch("svm", _RecordingModel(), ["x"], {"labels": LABELS})


def test_prediction_columns():
    probs = np.array([[0.1, 0.2, 0.7], [0.6, 0.3, 0.1]])
    cols = prediction_columns(probs, LABELS)
    assert list(cols) == ["prediction", "confidence", "prob_negative", "prob_neutral", "prob_positive"]
    assert cols["prediction"] == ["positive", "negative"]
    assert cols["confidence"] == [0.7, 0.6]
    assert cols["prob_neutral"] == [0.2, 0.3]


def test_prediction_columns_empty():
    cols = prediction_columns(np.zeros((0, 3)), LABELS)
    assert all(v == [] for v in cols.values())
