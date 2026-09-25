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
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

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
    return parser.parse_args()


def preprocess_split(split, map_emoticons):
    texts = [preprocess_tweet(t, map_emoticons=map_emoticons) for t in split["text"]]
    labels = np.array(split["label"])
    return texts, labels


def make_sequences(tokenizer, texts, max_len=MAX_LEN):
    seq = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seq, maxlen=max_len, padding="post", truncating="post")


def train_rnn(build_fn, out_dir, name, map_emoticons=False, epochs=EPOCHS):
    """Train a Keras RNN classifier and save tokenizer, best and final weights to out_dir."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    print("Loading dataset...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")

    print(f"Preprocessing (map_emoticons={map_emoticons})...")
    train_texts, train_labels = preprocess_split(ds["train"], map_emoticons)
    val_texts, val_labels = preprocess_split(ds["validation"], map_emoticons)
    test_texts, test_labels = preprocess_split(ds["test"], map_emoticons)

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

    vocab_size = min(MAX_VOCAB, len(tokenizer.word_index) + 1)
    print(f"Building {name} (vocab={vocab_size}, embed={EMBED_DIM}, max_len={MAX_LEN})")
    model = build_fn(vocab_size=vocab_size, max_len=MAX_LEN, embed_dim=EMBED_DIM)
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
        verbose=1,
    )

    # EarlyStopping restored the best weights, so model_final.keras holds the
    # same weights as best.keras. best.keras is kept for backward compatibility.
    final_path = out_dir / "model_final.keras"
    model.save(str(final_path))
    print("Saved final model:", final_path)

    print("Evaluating...")
    val_preds = np.argmax(model.predict(X_val, batch_size=BATCH), axis=1)
    test_preds = np.argmax(model.predict(X_test, batch_size=BATCH), axis=1)
    val_metrics = compute_metrics(val_labels, val_preds)
    test_metrics = compute_metrics(test_labels, test_preds)
    print("Validation metrics:", {k: v for k, v in val_metrics.items() if k in ("accuracy", "f1_macro")})
    print("Test metrics:", {k: v for k, v in test_metrics.items() if k in ("accuracy", "f1_macro")})
    return model, tokenizer, val_metrics, test_metrics
