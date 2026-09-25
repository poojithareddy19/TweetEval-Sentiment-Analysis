# Sentiment-Analysis

A comprehensive sentiment analysis application supporting multiple deep learning models including Logistic Regression, LSTM, GRU, and BERT.

## Features

- **Multiple Models**: Choose between LogisticRegression, LSTM, GRU, and BERT for sentiment classification
- **Pre-trained Models**: All models come pre-trained and ready to use
- **Easy-to-use Interface**: Streamlit-based web UI for text sentiment analysis
- **Fast Inference**: Optimized for quick predictions

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
ls -l models/lr/pipeline.joblib models/lstm/model_final.keras models/bert/model.safetensors
```

Real files are hundreds of KB to hundreds of MB. A pointer is a text file of about 130 bytes
that starts with `version https://git-lfs.github.com/spec/v1`. The app also detects pointers
and tells you to run `git lfs pull`.

### Running the Application

Start the Streamlit app:
```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

## Training

The training scripts import from the `src` package, so they must be run from the repository
root as modules. Running them as plain files (`python src/training/train_lr.py`) fails with
`ModuleNotFoundError: No module named 'src'`.

```bash
pip install -r requirements/train.txt

python -m src.training.train_lr      # TF-IDF + Logistic Regression, fast on CPU
python -m src.training.train_lstm    # BiLSTM, CPU is slow but works
python -m src.training.train_gru     # BiGRU
python -m src.training.train_bert    # RoBERTa fine-tuning, needs a GPU in practice
```

Each script overwrites the model files under `models/<name>/`. BERT checkpoints and
TensorBoard logs go to `outputs/bert_runs/`, which is gitignored.

## Usage

1. Open the Streamlit app in your browser
2. Select a model from the sidebar (LogisticRegression, LSTM, GRU, or BERT)
3. Enter text in the text area
4. Click "Predict" to analyze sentiment
5. View the sentiment classification and confidence scores

## Results

Metrics come from `results/<model>.json`, written by each training script on the official
TweetEval test split. Regenerate the table with:

```bash
python scripts/make_results_table.py
```

| Model | Accuracy | Macro F1 | Macro Recall |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.5922 | 0.5887 | 0.5987 |

Macro recall is the official TweetEval sentiment metric.

Note on the currently committed BERT model: according to its
`models/bert/checkpoint-17106/trainer_state.json` it reached 0.789 validation accuracy and
0.781 validation macro F1 at epoch 1 (the checkpoint that is shipped). Validation loss then
rose from 0.51 to 0.98 over epochs 2 and 3. That run started from
`cardiffnlp/twitter-roberta-base-sentiment`, a base that was already fine-tuned on this task,
so its numbers are not comparable with the other models. Retraining with `train_bert.py` now
starts from the task-neutral `cardiffnlp/twitter-roberta-base`. No test-split numbers exist for
any model until training is rerun.

## Supported Models

| Model | Type | Framework | File |
|-------|------|-----------|------|
| LogisticRegression | Shallow | scikit-learn | `models/lr/pipeline.joblib` |
| LSTM | Deep Learning | TensorFlow/Keras | `models/lstm/best.keras` (fallback `model_final.keras`) |
| GRU | Deep Learning | TensorFlow/Keras | `models/gru/best.keras` (fallback `model_final.keras`) |
| BERT | Transformer | Hugging Face | `models/bert/` (RoBERTa architecture) |

## Project Structure

```
Sentiment-Analysis/
├── app/
│   └── streamlit_app.py          # Main Streamlit application
├── models/
│   ├── bert/                      # Pre-trained BERT model
│   ├── gru/                       # Pre-trained GRU model
│   ├── lstm/                      # Pre-trained LSTM model
│   └── lr/                        # Pre-trained Logistic Regression model
├── src/
│   ├── models/                    # Model wrappers
│   ├── training/                  # Training scripts
│   └── utils/                     # Utility functions
└── requirements/
    ├── inference.txt              # Production dependencies
    └── train.txt                  # Training dependencies
```

## Dependencies

Main dependencies for inference:
- **transformers**: Hugging Face transformer models (for BERT)
- **torch**: PyTorch framework
- **tensorflow**: TensorFlow/Keras for LSTM and GRU
- **scikit-learn**: Machine learning utilities
- **streamlit**: Web UI framework
- **joblib**: Model serialization

## Notes

- All models are pre-trained and ready for inference
- No training data is required to run the application
- The app preprocesses input text automatically, following each model's `inference_config.json`
  (legacy defaults when the file is missing)
- Sentiment classes: Negative, Neutral, Positive

## License

See LICENSE file for details.

