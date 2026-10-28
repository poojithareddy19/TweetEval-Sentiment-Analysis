import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Ensure we can import from parent directory
current_dir = Path(__file__).parent
parent_dir = current_dir.parent
sys.path.insert(0, str(parent_dir))
os.chdir(str(parent_dir))

from src.models.batch import MAX_BATCH_ROWS, predict_proba_batch, prediction_columns  # noqa: E402
from src.models.inference import (  # noqa: E402
    keras_predict_proba,
    load_inference_config,
    load_keras_model,
    load_lr_pipeline,
    preprocess_for_model,
)

# -------------------------------------------------
# Streamlit Layout
# -------------------------------------------------
st.set_page_config(page_title="Sentiment Analysis", layout="centered")
st.title("Sentiment Analysis - LR, LSTM, GRU, RoBERTa")

MODEL_CHOICES = ["LogisticRegression", "LSTM", "GRU", "RoBERTa"]
MODEL_DIRS = {
    "LogisticRegression": "models/lr",
    "LSTM": "models/lstm",
    "GRU": "models/gru",
    "RoBERTa": "models/roberta",
}
model_choice = st.sidebar.selectbox("Select Model", MODEL_CHOICES)
mode = st.sidebar.radio(
    "Mode",
    ["Single text", "Batch CSV"],
    help="Batch CSV runs the selected model on one column of an uploaded CSV file.",
)
compare_all = mode == "Single text" and st.sidebar.checkbox(
    "Compare all models",
    value=False,
    help="Run every model whose files are available and show one table.",
)


# -------------------------------------------------
# Cached Loaders
# -------------------------------------------------

@st.cache_resource
def load_config(model_dir):
    """Per-model inference config; falls back to legacy defaults when the file is missing."""
    return load_inference_config(model_dir)


@st.cache_resource
def load_lr(model_dir="models/lr"):
    """Load Logistic Regression joblib pipeline."""
    return load_lr_pipeline(model_dir)


@st.cache_resource
def load_keras(model_dir):
    """Load tokenizer + Keras model (best.keras, else model_final.keras) for LSTM or GRU."""
    return load_keras_model(model_dir)


@st.cache_resource
def load_roberta(model_dir="models/roberta"):
    """Load the RoBERTa wrapper. Imported here so torch and transformers only load when RoBERTa is used."""
    from src.models.roberta_wrapper import RobertaWrapper

    return RobertaWrapper(model_dir)


# -------------------------------------------------
# Prediction
# -------------------------------------------------

def predict_proba(model_name, raw_text):
    """Preprocess raw_text as recorded in the model's config and return the softmax vector."""
    model_dir = MODEL_DIRS[model_name]
    cfg = load_config(model_dir)
    cleaned_text = preprocess_for_model(raw_text, cfg)

    if model_name == "LogisticRegression":
        model = load_lr(model_dir)
        return model.predict_proba([cleaned_text])[0], cfg
    if model_name in ("LSTM", "GRU"):
        tokenizer, model = load_keras(model_dir)
        return keras_predict_proba(tokenizer, model, cleaned_text, cfg), cfg
    roberta = load_roberta(model_dir)
    return roberta.predict_proba([cleaned_text])[0], cfg


def load_for_batch(model_name):
    """Return (kind, loaded model, config) for src.models.batch.predict_proba_batch."""
    model_dir = MODEL_DIRS[model_name]
    cfg = load_config(model_dir)
    if model_name == "LogisticRegression":
        return "lr", load_lr(model_dir), cfg
    if model_name in ("LSTM", "GRU"):
        return "keras", load_keras(model_dir), cfg
    return "roberta", load_roberta(model_dir), cfg


# -------------------------------------------------
# Batch CSV mode
# -------------------------------------------------

if mode == "Batch CSV":
    import pandas as pd

    st.write(f"Batch prediction with: **{model_choice}**")
    uploaded = st.file_uploader("Upload a CSV file", type=["csv"])
    if uploaded is None:
        st.stop()
    try:
        df = pd.read_csv(uploaded)
    except Exception as e:  # empty file, bad encoding, malformed CSV
        st.error(f"Could not read the CSV file: {e}")
        st.stop()
    if df.empty:
        st.warning("The CSV file has no rows.")
        st.stop()

    text_column = st.selectbox("Text column", list(df.columns))
    if len(df) > MAX_BATCH_ROWS:
        st.warning(f"The file has {len(df)} rows. Only the first {MAX_BATCH_ROWS} are used.")
        df = df.head(MAX_BATCH_ROWS)

    if st.button("Predict batch"):
        try:
            kind, model, cfg = load_for_batch(model_choice)
            probs = predict_proba_batch(kind, model, df[text_column].tolist(), cfg)
        except Exception as e:  # missing files, LFS pointers, load errors
            st.error(f"{model_choice} error: {e}")
            st.stop()
        result = df.copy()
        for name, values in prediction_columns(probs, cfg["labels"]).items():
            result[name] = values
        st.dataframe(result)
        st.download_button(
            "Download predictions as CSV",
            result.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
        )
    st.stop()


# -------------------------------------------------
# Main UI
# -------------------------------------------------

st.write("Comparing: **all available models**" if compare_all else f"Using Model: **{model_choice}**")
text = st.text_area("Enter text to analyze:", height=140)

if st.button("Predict"):
    if not text.strip():
        st.warning("Please enter some text.")
        st.stop()

    if compare_all:
        rows = []
        for name in MODEL_CHOICES:
            try:
                probs, cfg = predict_proba(name, text)
            except Exception as e:  # missing files, LFS pointers, load errors
                st.info(f"{name} skipped: {e}")
                continue
            labels = cfg["labels"]
            pred_idx = int(np.argmax(probs))
            row = {
                "Model": name,
                "Prediction": labels[pred_idx],
                "Confidence": round(float(probs[pred_idx]), 3),
            }
            row.update({f"P({label})": round(float(p), 3) for label, p in zip(labels, probs)})
            rows.append(row)
        if not rows:
            st.error("No model could be run. Check that the model files exist and are not Git LFS pointers.")
        else:
            st.table(rows)
        st.stop()

    try:
        probs, cfg = predict_proba(model_choice, text)
    except Exception as e:
        st.error(f"{model_choice} error: {e}")
        st.stop()

    labels = cfg["labels"]
    pred_idx = int(np.argmax(probs))

    st.subheader(f"Prediction: **{labels[pred_idx].upper()}**")
    st.write(f"Confidence: `{float(probs[pred_idx]):.3f}`")

    st.table({
        "Class": labels,
        "Probability": [float(p) for p in probs]
    })
