"""Per-model inference configuration and shared prediction helpers.

Each training script writes models/<name>/inference_config.json describing
how inputs must be preprocessed for that model. The app (and the smoke
tests) read it back. When the file is missing, the legacy behaviour the
committed weights were trained with is used: preprocess_tweet with default
arguments and max_len=80.
"""
import platform
from importlib import metadata
from pathlib import Path

from src.utils.io import check_not_lfs_pointer, load_json, save_json
from src.utils.preprocessing import preprocess_for_transformer, preprocess_tweet

CONFIG_FILENAME = "inference_config.json"
LABELS = ["negative", "neutral", "positive"]

# Behaviour of the weights committed before inference_config.json existed.
LEGACY_CONFIG = {
    "preprocessing": "tweet",
    "map_emoticons": False,
    "max_len": 80,
    "labels": LABELS,
    "versions": {},
    "source": "legacy-default",
}

_VERSION_PACKAGES = ["numpy", "scikit-learn", "joblib", "emoji", "tensorflow", "keras", "torch", "transformers"]


def library_versions():
    """Versions of the libraries relevant to loading a saved model (missing ones are skipped)."""
    versions = {"python": platform.python_version()}
    for name in _VERSION_PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return versions


def write_inference_config(model_dir, preprocessing, map_emoticons=False, max_len=None, extra=None):
    """Write models/<name>/inference_config.json and return the config dict."""
    cfg = {
        "preprocessing": preprocessing,
        "map_emoticons": bool(map_emoticons),
        "max_len": max_len,
        "labels": LABELS,
        "versions": library_versions(),
    }
    if extra:
        cfg.update(extra)
    save_json(cfg, Path(model_dir) / CONFIG_FILENAME)
    return cfg


def load_inference_config(model_dir):
    """Read the model's inference config, falling back to legacy defaults if absent."""
    path = Path(model_dir) / CONFIG_FILENAME
    if not path.exists():
        return dict(LEGACY_CONFIG)
    cfg = dict(LEGACY_CONFIG)
    cfg.update(load_json(path))
    cfg["source"] = str(path)
    return cfg


def preprocess_for_model(text, cfg):
    """Apply the preprocessing recorded in cfg to a raw input string."""
    if cfg.get("preprocessing") == "transformer":
        return preprocess_for_transformer(text)
    return preprocess_tweet(text, map_emoticons=bool(cfg.get("map_emoticons", False)))


def find_keras_model_path(model_dir):
    """Prefer best.keras, fall back to model_final.keras. Returns None if neither exists."""
    model_dir = Path(model_dir)
    for name in ("best.keras", "model_final.keras"):
        candidate = model_dir / name
        if candidate.exists():
            return candidate
    return None


def load_keras_model(model_dir):
    """Load tokenizer and Keras model from model_dir with LFS pointer checks."""
    from tensorflow.keras.models import load_model
    import joblib

    model_dir = Path(model_dir)
    tok_path = model_dir / "tokenizer.joblib"
    model_path = find_keras_model_path(model_dir)
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer.joblib missing in {model_dir}/")
    if model_path is None:
        raise FileNotFoundError(f"best.keras or model_final.keras missing in {model_dir}/")
    check_not_lfs_pointer(tok_path, "Keras tokenizer")
    check_not_lfs_pointer(model_path, "Keras model")
    return joblib.load(tok_path), load_model(str(model_path))


def keras_predict_proba(tokenizer, model, cleaned_text, cfg):
    """Tokenise, pad to cfg['max_len'] and return the softmax vector for one input."""
    from tensorflow.keras.preprocessing.sequence import pad_sequences

    max_len = int(cfg.get("max_len") or LEGACY_CONFIG["max_len"])
    seq = tokenizer.texts_to_sequences([cleaned_text])
    seq = pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")
    # Direct call instead of model.predict(): no tf.function tracing overhead for single inputs.
    return model(seq, training=False).numpy()[0]


def load_lr_pipeline(model_dir):
    """Load the scikit-learn pipeline with an LFS pointer check."""
    import joblib

    path = Path(model_dir) / "pipeline.joblib"
    if not path.exists():
        raise FileNotFoundError(f"pipeline.joblib missing in {model_dir}/")
    check_not_lfs_pointer(path, "LR pipeline")
    return joblib.load(path)
