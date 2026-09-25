"""Print a markdown results table from results/*.json for the README.

Usage (from the repo root):
    python scripts/make_results_table.py            # test split (default)
    python scripts/make_results_table.py --split validation
"""
import argparse
import json
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO / "results"
DISPLAY_NAMES = {"lr": "TF-IDF + Logistic Regression", "lstm": "BiLSTM", "gru": "BiGRU", "bert": "RoBERTa (fine-tuned)"}
ORDER = ["lr", "lstm", "gru", "bert"]


def build_table(results_dir=RESULTS_DIR, split="test"):
    rows = []
    files = {p.stem: p for p in sorted(Path(results_dir).glob("*.json"))}
    for key in ORDER + sorted(k for k in files if k not in ORDER):
        if key not in files:
            continue
        data = json.loads(files[key].read_text(encoding="utf-8"))
        metrics = data.get(split) or {}
        if not metrics:
            continue
        rows.append((
            DISPLAY_NAMES.get(key, key),
            metrics.get("accuracy"),
            metrics.get("f1_macro"),
            metrics.get("recall_macro"),
        ))

    header = "| Model | Accuracy | Macro F1 | Macro Recall |\n|---|---|---|---|"
    if not rows:
        return header + "\n| run training to populate | - | - | - |"

    def fmt(v):
        return "-" if v is None else f"{v:.4f}"

    body = "\n".join(f"| {name} | {fmt(a)} | {fmt(f)} | {fmt(r)} |" for name, a, f, r in rows)
    return header + "\n" + body


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test", choices=["test", "validation"])
    parser.add_argument("--results-dir", default=str(RESULTS_DIR))
    args = parser.parse_args()
    print(build_table(args.results_dir, args.split))


if __name__ == "__main__":
    main()
