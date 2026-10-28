"""Load each committed model and predict one sentence.

Skipped when the model files are missing or are unresolved Git LFS pointers,
so the suite still passes in CI where the weights are not checked out.
Set REQUIRE_MODELS=1 to turn every skip into a failure instead (used by the
models workflow, which checks out the weights).
"""
import importlib
import os
from pathlib import Path

import numpy as np
import pytest

from src.utils.io import is_lfs_pointer

REPO = Path(__file__).resolve().parent.parent
MODELS = REPO / "models"
SENTENCE = "I love this so much :)"
STRICT = os.environ.get("REQUIRE_MODELS") == "1"


def _skip(reason):
    if STRICT:
        pytest.fail(f"REQUIRE_MODELS=1: {reason}")
    pytest.skip(reason)


def _importorskip(name):
    if STRICT:
        try:
            return importlib.import_module(name)
        except ImportError as exc:
            pytest.fail(f"REQUIRE_MODELS=1: could not import {name}: {exc}")
    return pytest.importorskip(name)


def _require(*paths):
    for p in paths:
        if not p.exists():
            _skip(f"{p.relative_to(REPO)} not found")
        if is_lfs_pointer(p):
            _skip(f"{p.relative_to(REPO)} is a Git LFS pointer, run `git lfs pull`")


def _check_probs(probs):
    probs = np.asarray(probs, dtype=float)
    assert probs.shape == (3,)
    assert np.all(probs >= 0)
    assert probs.sum() == pytest.approx(1.0, abs=1e-4)


def test_lr_predicts():
    _require(MODELS / "lr" / "pipeline.joblib")
    from src.models.inference import load_inference_config, load_lr_pipeline, preprocess_for_model

    cfg = load_inference_config(MODELS / "lr")
    pipe = load_lr_pipeline(MODELS / "lr")
    probs = pipe.predict_proba([preprocess_for_model(SENTENCE, cfg)])[0]
    _check_probs(probs)


@pytest.mark.parametrize("name", ["lstm", "gru"])
def test_keras_rnn_predicts(name):
    _importorskip("tensorflow")
    from src.models.inference import (
        find_keras_model_path,
        keras_predict_proba,
        load_inference_config,
        load_keras_model,
        preprocess_for_model,
    )

    model_dir = MODELS / name
    model_path = find_keras_model_path(model_dir)
    if model_path is None:
        _skip(f"no .keras file in models/{name}/")
    _require(model_dir / "tokenizer.joblib", model_path)

    cfg = load_inference_config(model_dir)
    tok, model = load_keras_model(model_dir)
    probs = keras_predict_proba(tok, model, preprocess_for_model(SENTENCE, cfg), cfg)
    _check_probs(probs)


def test_roberta_predicts():
    _importorskip("torch")
    _importorskip("transformers")
    _require(MODELS / "roberta" / "model.safetensors", MODELS / "roberta" / "config.json")
    from src.models.inference import load_inference_config, preprocess_for_model
    from src.models.roberta_wrapper import RobertaWrapper

    cfg = load_inference_config(MODELS / "roberta")
    roberta = RobertaWrapper(str(MODELS / "roberta"), device="cpu")
    probs = roberta.predict_proba([preprocess_for_model(SENTENCE, cfg)])[0]
    _check_probs(probs)
