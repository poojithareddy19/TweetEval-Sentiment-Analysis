"""Evaluate the committed models on the official TweetEval sentiment splits.

Loads each model the same way the app does, applies the preprocessing from
its inference_config.json, predicts in batches and writes
<out-dir>/<model>.json (or <model>_<tag>.json) with the same validation/test
blocks as the training scripts.

Usage (from the repo root):
    python scripts/evaluate_models.py                          # all models, both splits
    python scripts/evaluate_models.py --models lr --subset 200 # smoke run
    python scripts/evaluate_models.py --models roberta --device cpu
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from src.models.inference import (  # noqa: E402
    find_keras_model_path,
    keras_predict_proba_batch,
    load_inference_config,
    load_keras_model,
    load_lr_pipeline,
    preprocess_for_model,
)
from src.utils.io import save_json  # noqa: E402
from src.utils.metrics import compute_metrics  # noqa: E402

MODELS = ["lr", "lstm", "gru", "roberta"]
SPLITS = ["validation", "test"]
KERAS_BATCH = 64


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    parser.add_argument("--splits", nargs="+", default=SPLITS, choices=SPLITS)
    parser.add_argument("--out-dir", default="results/eval")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only the first N examples of each split (smoke runs only).")
    parser.add_argument("--device", default=None, help="RoBERTa only: cpu or cuda (default: auto).")
    parser.add_argument("--batch-size", type=int, default=16, help="RoBERTa batch size.")
    parser.add_argument("--tag", default=None, help="Write <model>_<tag>.json instead of <model>.json.")
    parser.add_argument("--extra-json", default=None,
                        help="JSON object merged into the top level of the output, e.g. '{\"comparable\": false}'.")
    return parser.parse_args(argv)


def load_predictor(name, model_dir, device=None, batch_size=16):
    """Return (predict_proba(cleaned_texts) -> array, path of the weights file used)."""
    cfg = load_inference_config(model_dir)
    if name == "lr":
        pipe = load_lr_pipeline(model_dir)
        return (lambda texts: pipe.predict_proba(list(texts))), model_dir / "pipeline.joblib"
    if name in ("lstm", "gru"):
        tok, model = load_keras_model(model_dir)
        return (lambda texts: keras_predict_proba_batch(tok, model, texts, cfg, batch_size=KERAS_BATCH),
                find_keras_model_path(model_dir))
    from src.models.roberta_wrapper import RobertaWrapper

    roberta = RobertaWrapper(str(model_dir), device=device)
    max_len = int(cfg.get("max_len") or 128)
    return (lambda texts: roberta.predict_proba(list(texts), max_length=max_len, batch_size=batch_size),
            model_dir / "model.safetensors")


def evaluate(name, ds, splits, subset=None, device=None, batch_size=16):
    model_dir = REPO / "models" / name
    cfg = load_inference_config(model_dir)
    predict, weights_path = load_predictor(name, model_dir, device=device, batch_size=batch_size)
    result = {
        "model": name,
        "model_path": weights_path.relative_to(REPO).as_posix(),
        "config_source": Path(cfg["source"]).relative_to(REPO).as_posix() if cfg["source"] != "legacy-default"
        else cfg["source"],
        "evaluated_with": "scripts/evaluate_models.py",
        "subset": subset,
    }
    for split in splits:
        rows = ds[split]
        if subset:
            rows = rows.select(range(min(subset, len(rows))))
        texts = [preprocess_for_model(t, cfg) for t in rows["text"]]
        start = time.perf_counter()
        probs = np.asarray(predict(texts))
        elapsed = time.perf_counter() - start
        result[split] = compute_metrics(rows["label"], np.argmax(probs, axis=1))
        m = result[split]
        print(f"{name} {split}: n={len(texts)} acc={m['accuracy']:.4f} f1={m['f1_macro']:.4f} "
              f"recall={m['recall_macro']:.4f} ({elapsed:.1f}s)")
    return result


def main(argv=None):
    args = parse_args(argv)
    extra = json.loads(args.extra_json) if args.extra_json else {}
    if not isinstance(extra, dict):
        raise SystemExit("--extra-json must be a JSON object")

    from datasets import load_dataset

    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    out_dir = Path(args.out_dir)
    for name in args.models:
        result = evaluate(name, ds, args.splits, args.subset, args.device, args.batch_size)
        if args.tag:
            result["tag"] = args.tag
        result.update(extra)
        stem = f"{name}_{args.tag}" if args.tag else name
        path = out_dir / f"{stem}.json"
        save_json(result, path)
        print("Saved", path)


if __name__ == "__main__":
    main()
