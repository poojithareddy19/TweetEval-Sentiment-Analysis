"""Batch prediction for many raw texts, independent of Streamlit.

Used by the app's batch CSV mode. Inputs go through the same
preprocess_for_model() as single predictions, so non-string cells such as
NaN from a CSV become "".
"""
import numpy as np

from src.models.inference import keras_predict_proba_batch, preprocess_for_model

MAX_BATCH_ROWS = 1000


def predict_proba_batch(kind, model, raw_texts, cfg):
    """Return an (n, n_labels) array of probabilities for raw_texts.

    kind is "lr" (scikit-learn pipeline), "keras" ((tokenizer, model) tuple)
    or "roberta" (RobertaWrapper).
    """
    n_labels = len(cfg["labels"])
    cleaned = [preprocess_for_model(t, cfg) for t in raw_texts]
    if not cleaned:
        return np.zeros((0, n_labels))
    if kind == "lr":
        return np.asarray(model.predict_proba(cleaned))
    if kind == "keras":
        tokenizer, keras_model = model
        return np.asarray(keras_predict_proba_batch(tokenizer, keras_model, cleaned, cfg))
    if kind == "roberta":
        return np.asarray(model.predict_proba(cleaned))
    raise ValueError(f"unknown model kind: {kind!r}")


def prediction_columns(probs, labels):
    """Column name -> list of values: prediction, confidence and prob_<label> per label."""
    probs = np.asarray(probs, dtype=float).reshape(-1, len(labels))
    idx = np.argmax(probs, axis=1) if len(probs) else np.zeros(0, dtype=int)
    columns = {
        "prediction": [labels[i] for i in idx],
        "confidence": [float(probs[row, i]) for row, i in enumerate(idx)],
    }
    for j, label in enumerate(labels):
        columns[f"prob_{label}"] = probs[:, j].tolist()
    return columns
