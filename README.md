# Sentiment-Analysis

A comprehensive sentiment analysis application supporting a classical baseline and deep learning models: Logistic Regression, LSTM, GRU, and RoBERTa.

## Features

- **Multiple Models**: Choose between LogisticRegression, LSTM, GRU, and RoBERTa for sentiment classification
- **Pre-trained Models**: All models come pre-trained and ready to use after `git lfs pull`
- **Easy-to-use Interface**: Streamlit-based web UI for text sentiment analysis
- **Batch CSV Mode**: Predict a column of an uploaded CSV file and download the results
- **Model Caching**: Each model is loaded once and cached for later predictions
- **Lazy PyTorch Import**: PyTorch and transformers are only imported when RoBERTa is used

## Setup Instructions

### Prerequisites

- Python 3.10 or 3.11
- pip or conda package manager
- Git and Git LFS (the model weights are stored with Git LFS)

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/poojithareddy19/TweetEval-Sentiment-Analysis.git
cd TweetEval-Sentiment-Analysis
```

2. **Create a virtual environment (optional but recommended)**:
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n sentiment-analysis python=3.10
conda activate sentiment-analysis
```

3. **Install dependencies**:
```bash
pip install -r requirements/inference.txt
```

4. **Fetch the model weights with Git LFS**:

The trained models under `models/` are stored with Git LFS. Without LFS a clone only
contains small pointer files and the app fails with cryptic pickle or safetensors errors.

```bash
# install Git LFS first: https://git-lfs.com (e.g. apt install git-lfs, brew install git-lfs,
# or the Windows installer), then:
git lfs install
git lfs pull
```

Check that the files are real binaries and not pointers:

```bash
git lfs ls-files
ls -l models/lr/pipeline.joblib models/lstm/model_final.keras models/roberta/model.safetensors
```

Real files are hundreds of KB to hundreds of MB. A pointer is a text file of about 130 bytes
that starts with `version https://git-lfs.github.com/spec/v1`. The app also detects pointers
and tells you to run `git lfs pull`.

### Running the Application

Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

Alternatively, run `python run_app.py` from the repository root, which launches the same app.

The app will be available at `http://localhost:8501`

### Docker

The `Dockerfile` runs the Streamlit app with all four models, for local use. Fetch the weights
with `git lfs pull` first: the build stops with a list of files if any model file is still an
LFS pointer (`scripts/check_lfs_pointers.py`).

```bash
docker build -t tweeteval-sentiment .
docker run --rm -p 8501:8501 tweeteval-sentiment
```

The app is then at `http://localhost:8501`, and `http://localhost:8501/_stcore/health` returns
`ok` (the container's `HEALTHCHECK` uses it). PyTorch is installed from the CPU-only index at
the version pinned in `requirements/inference.txt`, so the container does not use a GPU. The
image is about 6 GB, mostly TensorFlow, PyTorch and the 550 MB of model files.

### Batch predictions

In the sidebar, set **Mode** to **Batch CSV**, upload a CSV file and choose the text column. The
selected model predicts every row (up to the first 1,000) and the results can be downloaded as a
CSV with `prediction`, `confidence` and one `prob_<label>` column per class.

## Running tests

```bash
pip install -r requirements/dev.txt
pytest
ruff check .
```

The model smoke tests in `tests/test_models_smoke.py` are skipped when the weights are missing
or are Git LFS pointers. Set `REQUIRE_MODELS=1` to make them fail instead, as the manual
`models` GitHub workflow does.

## Training

The training scripts import from the `src` package, so they must be run from the repository
root as modules. Running them as plain files (`python src/training/train_lr.py`) fails with
`ModuleNotFoundError: No module named 'src'`.

```bash
pip install -r requirements/train.txt

python -m src.training.train_lr      # TF-IDF + Logistic Regression, fast on CPU
python -m src.training.train_lstm    # BiLSTM, CPU is slow but works
python -m src.training.train_gru     # BiGRU
python -m src.training.train_roberta # RoBERTa fine-tuning, needs a GPU in practice
```

Each script overwrites the model files under `models/<name>/`. RoBERTa checkpoints and
TensorBoard logs go to `outputs/roberta_runs/`, which is gitignored.

## Usage

1. Open the Streamlit app in your browser
2. Select a model from the sidebar (LogisticRegression, LSTM, GRU, or RoBERTa)
3. Enter text in the text area
4. Click "Predict" to analyze sentiment
5. View the sentiment classification and confidence scores

## Results

Metrics come from `results/<model>.json` on the official TweetEval test split, written by each
training script (or, for the legacy RoBERTa row, by `scripts/evaluate_models.py`). Regenerate the
table with:

```bash
python scripts/make_results_table.py
```

| Model | Accuracy | Macro F1 | Macro Recall |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.5922 | 0.5887 | 0.5987 |
| BiLSTM | 0.5785 | 0.5726 | 0.5841 |
| BiGRU | 0.5693 | 0.5641 | 0.5799 |
| RoBERTa (legacy, task-tuned base, not comparable) | 0.7142 | 0.7131 | 0.7179 |

Macro recall is the official TweetEval sentiment metric. The BiLSTM and BiGRU rows come from
`python -m src.training.train_lstm --map-emoticons` and the GRU equivalent (10 epochs max,
early stopping patience 2, balanced class weights, both stopped after epoch 4 with epoch 2
restored). No comparable RoBERTa row exists yet because retraining from the task-neutral base
needs a GPU. The last row is the committed legacy RoBERTa model (see below).

Note on the currently committed RoBERTa model: according to its
`results/roberta_trainer_state_2025-12-05.json` (the Trainer log preserved from the original run) it reached 0.789 validation accuracy and
0.781 validation macro F1 at epoch 1 (the checkpoint that is shipped). Validation loss then
rose from 0.51 to 0.98 over epochs 2 and 3. That run started from
`cardiffnlp/twitter-roberta-base-sentiment`, a base that was already fine-tuned on this task,
so its numbers are not comparable with the other models. Retraining with `train_roberta.py` now
starts from the task-neutral `cardiffnlp/twitter-roberta-base`. The committed model's test-split
numbers (last table row, `results/roberta_legacy.json`, produced with
`python scripts/evaluate_models.py --models roberta --out-dir results --tag legacy ...`) are
shown for reference only and are not comparable with the other rows.

## Supported Models

| Model | Type | Framework | File |
|-------|------|-----------|------|
| LogisticRegression | Shallow | scikit-learn | `models/lr/pipeline.joblib` |
| LSTM | Deep Learning | TensorFlow/Keras | `models/lstm/best.keras` (fallback `model_final.keras`) |
| GRU | Deep Learning | TensorFlow/Keras | `models/gru/best.keras` (fallback `model_final.keras`) |
| RoBERTa | Transformer | Hugging Face | `models/roberta/` |

## Project Structure

```
TweetEval-Sentiment-Analysis/
├── .github/
│   └── workflows/                 # CI (tests.yml) and manual real-model run (models.yml)
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── models/
│   ├── roberta/                   # Fine-tuned RoBERTa model
│   ├── gru/                       # Pre-trained GRU model
│   ├── lstm/                      # Pre-trained LSTM model
│   └── lr/                        # Pre-trained Logistic Regression model
├── src/
│   ├── models/                    # Model wrappers
│   ├── training/                  # Training scripts
│   └── utils/                     # Utility functions
├── results/                       # Metrics JSON written by the training scripts
├── scripts/
│   ├── check_lfs_pointers.py      # Fails if a model file is a Git LFS pointer
│   ├── evaluate_models.py         # Evaluates the committed models on TweetEval
│   └── make_results_table.py      # Builds the README results table
├── tests/                         # pytest suite
├── requirements/
│   ├── inference.txt              # Production dependencies
│   ├── train.txt                  # Training dependencies
│   └── dev.txt                    # Test and lint dependencies
├── Dockerfile                     # Container for running the app locally
├── pyproject.toml                 # pytest and ruff configuration
└── run_app.py                     # Launches the Streamlit app from the repo root
```

## Dependencies

Main dependencies for inference:
- **transformers**: Hugging Face transformer models (for RoBERTa)
- **torch**: PyTorch framework
- **tensorflow**: TensorFlow/Keras for LSTM and GRU
- **scikit-learn**: Machine learning utilities
- **streamlit**: Web UI framework
- **joblib**: Model serialization

## Notes

- All models are pre-trained and ready for inference after `git lfs pull`
- No training data is required to run the application
- The app preprocesses input text automatically, following each model's `inference_config.json`
  (legacy defaults when the file is missing)
- Sentiment classes: Negative, Neutral, Positive

## License

See LICENSE file for details.

