# src/training/train_lr.py
import argparse
from pathlib import Path

import joblib
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

from src.models.inference import write_inference_config
from src.utils.io import save_json
from src.utils.metrics import compute_metrics
from src.utils.preprocessing import preprocess_tweet

OUT_DIR = Path("models/lr")
RESULTS_PATH = Path("results/lr.json")


def parse_args():
    parser = argparse.ArgumentParser(description="Train the TF-IDF + Logistic Regression model on TweetEval.")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Where to save pipeline.joblib and the config.")
    parser.add_argument("--results-path", default=str(RESULTS_PATH), help="Where to save the metrics JSON.")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only the first N examples of each split (smoke runs only).")
    return parser.parse_args()


def ds_to_df(ds, split, subset=None):
    rows = ds[split]
    if subset:
        rows = rows.select(range(min(subset, len(rows))))
    texts = [preprocess_tweet(t) for t in rows['text']]
    labels = rows['label']
    return pd.DataFrame({"text": texts, "label": labels})

def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = Path(args.results_path)

    print("Loading dataset...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    train_df = ds_to_df(ds, "train", args.subset)
    val_df = ds_to_df(ds, "validation", args.subset)
    test_df = ds_to_df(ds, "test", args.subset)

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1,2), min_df=3, max_df=0.9)),
        ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='saga'))
    ])

    # Small but real grid. The previous grid had a single combination, which
    # only paid the cross-validation cost without searching anything.
    params = {
        'tfidf__max_features': [20000, 50000, None],
        'clf__C': [0.5, 1.0, 2.0, 4.0],
    }

    print("Starting GridSearchCV (this may take a while)...")
    gs = GridSearchCV(pipe, params, cv=3, scoring='f1_macro', n_jobs=-1, verbose=1)
    gs.fit(train_df['text'], train_df['label'])

    best = gs.best_estimator_
    print("Best params:", gs.best_params_)

    val_preds = best.predict(val_df['text'])
    test_preds = best.predict(test_df['text'])

    val_metrics = compute_metrics(val_df['label'], val_preds)
    test_metrics = compute_metrics(test_df['label'], test_preds)
    print("Validation:", {k: v for k, v in val_metrics.items() if k in ("accuracy", "f1_macro", "recall_macro")})
    print("Test:", {k: v for k, v in test_metrics.items() if k in ("accuracy", "f1_macro", "recall_macro")})
    save_json({"model": "lr", "best_params": gs.best_params_, "subset": args.subset,
               "validation": val_metrics, "test": test_metrics}, results_path)
    print("Saved results to:", results_path)

    model_path = out_dir / "pipeline.joblib"
    joblib.dump(best, model_path)
    print(f"Saved LR pipeline to: {model_path}")

    write_inference_config(out_dir, preprocessing="tweet", map_emoticons=False,
                           extra={"best_params": gs.best_params_})
    print("Saved inference config to:", out_dir / "inference_config.json")

if __name__ == "__main__":
    main()
