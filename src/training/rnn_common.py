"""Shared training loop for the BiLSTM and BiGRU models.

train_lstm.py and train_gru.py are thin entry points that call train_rnn()
with their own build function and output directory.
"""
import argparse
from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf
from datasets import load_dataset
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from src.models.inference import write_inference_config
from src.utils.io import save_json
from src.utils.metrics import compute_metrics
from src.utils.preprocessing import preprocess_tweet

# ===========================
# FIXED HYPERPARAMETERS
# ===========================
MAX_VOCAB = 10000
MAX_LEN = 80
EMBED_DIM = 100
BATCH = 64
EPOCHS = 10
PATIENCE = 2
SEED = 42
NUM_CLASSES = 3
# ===========================


def parse_args(name):
    parser = argparse.ArgumentParser(description=f"Train the Bi{name} sentiment model on TweetEval.")
    parser.add_argument(
        "--map-emoticons",
        action="store_true",
        help="Map text emoticons such as :) and :( to word tokens before tokenising. Opt-in; "
        "recorded in inference_config.json so the app applies the same preprocessing.",
    )
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--out-dir", default=None, help="Override the model output directory (smoke runs).")
    parser.add_argument("--results-path", default=None, help="Override the metrics JSON path (smoke runs).")
    parser.add_argument("--subset", type=int, default=None,
                        help="Use only the first N examples of each split (smoke runs only).")
    parser.add_argument(
        "--glove-path",
        default=None,
        help="Optional path to a GloVe text file (e.g. glove.twitter.27B.100d.txt) used to initialise "
        "the embedding layer. Its dimension must equal EMBED_DIM. Not downloaded automatically.",
    )
    return parser.parse_args()


def load_glove_embeddings(glove_path, tokenizer, vocab_size, embed_dim):
    """Build a (vocab_size, embed_dim) matrix from a GloVe text file for the tokenizer's vocabulary.

    Words missing from the GloVe file keep a small random init (Keras' default
    uniform(-0.05, 0.05)); row 0 (padding) is zero.
    """
    rng = np.random.default_rng(SEED)
    matrix = rng.uniform(-0.05, 0.05, size=(vocab_size, embed_dim)).astype("float32")
    matrix[0] = 0.0
    found = 0
    with open(glove_path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            parts = line.rstrip().split(" ")
            idx = tokenizer.word_index.get(parts[0])
            if idx is None or idx >= vocab_size:
                continue
            vec = np.asarray(parts[1:], dtype="float32")
            if vec.shape[0] != embed_dim:
                raise ValueError(
                    f"GloVe file has dimension {vec.shape[0]} but EMBED_DIM is {embed_dim}; "
                    "use the matching GloVe file or change EMBED_DIM."
                )
            matrix[idx] = vec
            found += 1
    print(f"GloVe coverage: {found}/{vocab_size - 1} vocabulary words found in {glove_path}")
    return matrix


def preprocess_split(split, map_emoticons, subset=None):
    if subset:
        split = split.select(range(min(subset, len(split))))
    texts = [preprocess_tweet(t, map_emoticons=map_emoticons) for t in split["text"]]
    labels = np.array(split["label"])
    return texts, labels


def make_sequences(tokenizer, texts, max_len=MAX_LEN):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")


def train_rnn(build_fn, out_dir, name, map_emoticons=False, epochs=EPOCHS, glove_path=None,
              results_path=None, subset=None):
    """Train a Keras RNN classifier and save tokenizer, best and final weights to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("Loading dataset...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    print(f"Preprocessing (map_emoticons={map_emoticons})...")
    train_texts, train_labels = preprocess_split(ds["train"], map_emoticons, subset)
    val_texts, val_labels = preprocess_split(ds["validation"], map_emoticons, subset)
    test_texts, test_labels = preprocess_split(ds["test"], map_emoticons, subset)

    print(f"Fitting tokenizer (max_vocab={MAX_VOCAB})...")
    tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
    tokenizer.fit_on_texts(train_texts)
    tok_path = out_dir / "tokenizer.joblib"
    joblib.dump(tokenizer, tok_path)
    print("Saved tokenizer:", tok_path)

    print("Converting to sequences...")
    X_train = make_sequences(tokenizer, train_texts)
    X_val = make_sequences(tokenizer, val_texts)
    X_test = make_sequences(tokenizer, test_texts)

    y_train = tf.keras.utils.to_categorical(train_labels, NUM_CLASSES)
    y_val = tf.keras.utils.to_categorical(val_labels, NUM_CLASSES)

    # Balanced class weights: TweetEval sentiment is skewed towards neutral.
    class_weights = compute_class_weight("balanced", classes=np.arange(NUM_CLASSES), y=train_labels)
    class_weight = {int(i): float(w) for i, w in enumerate(class_weights)}
    print("Class weights (balanced):", class_weight)

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    print(f"Building {name} (vocab={vocab_size}, embed={EMBED_DIM}, max_len={MAX_LEN})")
    embedding_weights = None
    if glove_path:
        embedding_weights = load_glove_embeddings(glove_path, tokenizer, vocab_size, EMBED_DIM)
    model = build_fn(vocab_size=vocab_size, max_len=MAX_LEN, embed_dim=EMBED_DIM, embedding_weights=embedding_weights)
    model.summary()

    ckpt = ModelCheckpoint(str(out_dir / "best.keras"), save_best_only=True, monitor="val_loss")
    es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True)

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=BATCH,
        callbacks=[ckpt, es],
        class_weight=class_weight,
        verbose=1,
    )

    # EarlyStopping restored the best weights, so model_final.keras holds the
    # same weights as best.keras. best.keras is kept for backward compatibility.
    final_path = out_dir / "model_final.keras"
    model.save(str(final_path))
    print("Saved final model:", final_path)

    write_inference_config(out_dir, preprocessing="tweet", map_emoticons=map_emoticons, max_len=MAX_LEN,
                           extra={"max_vocab": MAX_VOCAB, "embed_dim": EMBED_DIM,
                                  "glove_file": Path(glove_path).name if glove_path else None})
    print("Saved inference config to:", out_dir / "inference_config.json")

    print("Evaluating...")
    val_preds = np.argmax(model.predict(X_val, batch_size=BATCH), axis=1)
    test_preds = np.argmax(model.predict(X_test, batch_size=BATCH), axis=1)
    val_metrics = compute_metrics(val_labels, val_preds)
    test_metrics = compute_metrics(test_labels, test_preds)
    summary_keys = ("accuracy", "f1_macro", "recall_macro")
    print("Validation metrics:", {k: v for k, v in val_metrics.items() if k in summary_keys})
    print("Test metrics:", {k: v for k, v in test_metrics.items() if k in summary_keys})
    results_path = Path(results_path) if results_path else Path("results") / f"{name.lower()}.json"
    save_json({"model": name.lower(), "map_emoticons": map_emoticons, "epochs_max": epochs, "subset": subset,
               "class_weight": class_weight, "glove_file": Path(glove_path).name if glove_path else None,
               "validation": val_metrics, "test": test_metrics}, results_path)
    print("Saved results to:", results_path)
    return model, tokenizer, val_metrics, test_metrics
