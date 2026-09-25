import argparse
import os
from pathlib import Path

# This script is PyTorch-only. tensorflow is installed in the same environment for
# the Keras models, and with Keras 3 transformers' TensorFlow integration fails to
# import ("install tf-keras"). USE_TF=0 makes transformers ignore TensorFlow. It must
# be set before transformers is imported.
os.environ.setdefault("USE_TF", "0")

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from sklearn.metrics import accuracy_score, f1_score, recall_score  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.models.inference import write_inference_config  # noqa: E402
from src.utils.io import save_json  # noqa: E402
from src.utils.metrics import compute_metrics as full_metrics  # noqa: E402
from src.utils.preprocessing import preprocess_for_transformer  # noqa: E402

# Base model. This must be a model that has NOT been fine-tuned on TweetEval
# sentiment, otherwise the comparison with LR, LSTM and GRU is not fair.
# cardiffnlp/twitter-roberta-base is pretrained on tweets only (masked LM),
# while cardiffnlp/twitter-roberta-base-sentiment (the previous choice) was
# already fine-tuned on this exact task.
MODEL_NAME = "cardiffnlp/twitter-roberta-base"
OUT_DIR = "models/bert"          # final model + tokenizer only
RUN_DIR = "outputs/bert_runs"    # checkpoints and TensorBoard logs (gitignored)
RESULTS_PATH = "results/bert.json"
NUM_LABELS = 3
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}
BATCH_SIZE = 8
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
EARLY_STOPPING_PATIENCE = 1
MAX_LEN = 128
SEED = 42
FP16 = False
# ---------------------------

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_examples(examples):
    texts = [preprocess_for_transformer(t) for t in examples["text"]]
    return {"text": texts, "label": examples["label"]}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
        # Macro recall is the official TweetEval sentiment metric.
        "recall_macro": recall_score(labels, preds, average="macro"),
    }

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a RoBERTa model on TweetEval sentiment.")
    parser.add_argument("--out-dir", default=OUT_DIR, help="Where to save the final model and tokenizer.")
    parser.add_argument("--run-dir", default=RUN_DIR, help="Where checkpoints and logs go.")
    parser.add_argument("--results-path", default=RESULTS_PATH, help="Where to save the metrics JSON.")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only the first N examples of each split (smoke runs only).")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir, run_dir, results_path = args.out_dir, args.run_dir, args.results_path
    set_seed(SEED)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(run_dir).mkdir(parents=True, exist_ok=True)

    print("Loading dataset (tweet_eval: sentiment)...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
    if args.subset:
        for split in list(ds.keys()):
            ds[split] = ds[split].select(range(min(args.subset, len(ds[split]))))

    ds = ds.map(preprocess_examples, batched=True)

    print("Loading tokenizer & model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=MAX_LEN)

    print("Tokenizing dataset...")
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=run_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=args.epochs,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="recall_macro",
        greater_is_better=True,
        fp16=FP16,
        seed=SEED,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)],
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on validation and test sets...")
    results = {"model": "bert", "base_model": MODEL_NAME, "subset": args.subset}
    for split in ("validation", "test"):
        out = trainer.predict(tokenized[split])
        preds = np.argmax(out.predictions, axis=-1)
        results[split] = full_metrics(out.label_ids, preds)
        summary = {k: v for k, v in results[split].items() if k in ("accuracy", "f1_macro", "recall_macro")}
        print(f"{split} metrics:", summary)
    save_json(results, results_path)
    print("Saved results to", results_path)

    print("Saving model & tokenizer to", out_dir)
    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)
    write_inference_config(out_dir, preprocessing="transformer", map_emoticons=False, max_len=MAX_LEN,
                           extra={"base_model": MODEL_NAME})
    print("Saved. Use src/models/bert_wrapper.py to run inference.")

if __name__ == "__main__":
    main()
