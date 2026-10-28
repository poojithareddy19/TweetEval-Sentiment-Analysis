# TweetEval Sentiment Analysis: Project Documentation

This document describes the models, their measured results, known limitations, testing and
deployment. Every metric below comes from a JSON file in `results/` written by a script
(`src/training/*.py` or `scripts/evaluate_models.py`). The audit that produced the current state
of the repository, with command output for each claim, is in `AUDIT_REPORT.md`.

## Overview

The project classifies tweets as negative, neutral or positive and serves four models through a
Streamlit app:

| Model | Type | Files |
|---|---|---|
| TF-IDF + Logistic Regression | Classical baseline (scikit-learn) | `models/lr/pipeline.joblib` |
| BiLSTM | Recurrent network (Keras) | `models/lstm/best.keras`, `tokenizer.joblib` |
| BiGRU | Recurrent network (Keras) | `models/gru/best.keras`, `tokenizer.joblib` |
| RoBERTa | Transformer (Hugging Face, PyTorch) | `models/roberta/` |

Each model directory has an `inference_config.json` recording the preprocessing, maximum length
and labels the model was trained with. The app and the evaluation script read it, so inputs are
always preprocessed the way the model expects.

## Data

All models use the official splits of `cardiffnlp/tweet_eval`, configuration `sentiment`
(label 0 negative, 1 neutral, 2 positive):

| Split | Tweets | Negative | Neutral | Positive |
|---|---|---|---|---|
| train | 45,615 | 7,093 (15.5%) | 20,673 (45.3%) | 17,849 (39.1%) |
| validation | 2,000 | 312 (15.6%) | 869 (43.5%) | 819 (40.9%) |
| test | 12,284 | 3,972 (32.3%) | 5,937 (48.3%) | 2,375 (19.3%) |

The test split has twice the share of negative tweets and half the share of positive tweets of
the training and validation splits.

## Models

- **Logistic Regression**: TF-IDF over word unigrams and bigrams (`min_df=3`, `max_df=0.9`),
  `class_weight="balanced"`, grid search over `C` and `max_features` with 3-fold CV on macro F1.
  Selected: `C=2.0`, no feature cap. Input: `preprocess_tweet` defaults.
- **BiLSTM / BiGRU**: embedding (vocabulary 10,000, dimension 100, trained from scratch), two
  bidirectional recurrent layers (128 and 64 units) with dropout, dense layer, softmax. Up to 10
  epochs, early stopping on validation loss with patience 2 and best weights restored, balanced
  class weights, emoticon mapping (`--map-emoticons`), sequences padded and truncated to 80 tokens.
- **RoBERTa (legacy)**: fine-tuned from `cardiffnlp/twitter-roberta-base-sentiment`, a model that
  was already fine-tuned on TweetEval sentiment. Input: `preprocess_tweet` defaults, 128 tokens.
  The shipped weights are the epoch 1 checkpoint of a 3-epoch run
  (`results/roberta_trainer_state_2025-12-05.json`). `src/training/train_roberta.py` now starts
  from the task-neutral `cardiffnlp/twitter-roberta-base` instead, but that retrain has not been
  run.

## Results

Macro recall is the official TweetEval sentiment metric. Tables are the output of
`python scripts/make_results_table.py` (test) and `--split validation`.

Test split:

| Model | Accuracy | Macro F1 | Macro Recall |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.5922 | 0.5887 | 0.5987 |
| BiLSTM | 0.5785 | 0.5726 | 0.5841 |
| BiGRU | 0.5693 | 0.5641 | 0.5799 |
| RoBERTa (legacy, task-tuned base, not comparable) | 0.7142 | 0.7131 | 0.7179 |

Validation split:

| Model | Accuracy | Macro F1 | Macro Recall |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.6660 | 0.6434 | 0.6595 |
| BiLSTM | 0.6145 | 0.6053 | 0.6579 |
| BiGRU | 0.6145 | 0.6036 | 0.6556 |
| RoBERTa (legacy, task-tuned base, not comparable) | 0.7890 | 0.7806 | 0.7824 |

The RoBERTa row is shown for reference only. Its base model had already been trained on this
task, so it is not a fair comparison with the other three.

Reproducibility: `python scripts/evaluate_models.py --models lr lstm gru` re-evaluates the
committed weights and reproduces `results/lr.json`, `lstm.json` and `gru.json` exactly
(identical metrics and confusion matrices); its output is kept in `results/eval/`. For RoBERTa,
the re-evaluated validation macro F1 equals the Trainer log's `best_metric`.

Per-class test results (from the same JSON files):

| Model | Recall negative / neutral / positive | Neutral tweets predicted negative |
|---|---|---|
| TF-IDF + Logistic Regression | 0.6289 / 0.5629 / 0.6042 | 1,658 of 5,937 (27.9%) |
| BiLSTM | 0.7729 / 0.4800 / 0.4994 | 2,475 of 5,937 (41.7%) |
| BiGRU | 0.7998 / 0.4447 / 0.4952 | 2,681 of 5,937 (45.2%) |
| RoBERTa (legacy, not comparable) | 0.8155 / 0.6613 / 0.6771 | 1,442 of 5,937 (24.3%) |

## Limitations

- **No comparable transformer result.** The only RoBERTa model was fine-tuned from a base that
  already knew the task. Retraining from the task-neutral base needs a CUDA GPU.
- **Validation-to-test drop.** Every model scores lower on test than on validation (LR macro
  recall 0.6595 to 0.5987, BiLSTM 0.6579 to 0.5841). The label distribution shift shown under
  Data is one likely contributor; it has not been isolated experimentally.
- **RNN neutral-to-negative confusion.** BiLSTM and BiGRU label 41.7% and 45.2% of neutral test
  tweets as negative. An ablation (BiLSTM trained with and without class weights, details in
  `AUDIT_REPORT.md`, Phase 5) showed the balanced class weights drive this, but removing them
  shifts errors to negative tweets predicted as neutral and does not improve macro recall, so the
  committed models were kept.
- **Limited tuning.** The RNNs use fixed hyperparameters and embeddings trained from scratch;
  `--glove-path` exists but has not been used for a committed model.
- **Heavy runtime.** Running all four models needs both TensorFlow and PyTorch. The Docker image
  is about 6 GB.
- **Training scope.** Only TweetEval sentiment is used; behaviour on other domains is untested.
- **English tweets only.** Preprocessing lowercases, replaces URLs and mentions, strips `#` from
  hashtags and converts emoji to text names (`emoji.demojize`); it is tuned for English Twitter
  text.

## Future Enhancements

- Retrain RoBERTa from `cardiffnlp/twitter-roberta-base` on a GPU and add a comparable row.
- Hyperparameter search for the RNNs, and a run with Twitter GloVe embeddings
  (`--glove-path`, 100-dimensional file).
- Address the neutral-to-negative confusion with a method that does not trade it for the
  opposite error, for example tuning class weights between none and fully balanced, or
  threshold adjustment on validation data.
- Repeat training runs with several seeds to measure run-to-run variance.
- Smaller deployment images, for example one per framework.

## Testing

```bash
pip install -r requirements/dev.txt
pytest
ruff check .
```

| File | Covers |
|---|---|
| `tests/test_preprocessing.py` | `preprocess_tweet` defaults stay identical to the version the models were trained with (`_legacy`), emoticon mapping, transformer preprocessing |
| `tests/test_inference_config.py` | RoBERTa inference config is loaded from its file and matches `preprocess_tweet` |
| `tests/test_io.py` | JSON helpers and Git LFS pointer detection |
| `tests/test_metrics.py` | `compute_metrics` |
| `tests/test_results_table.py` | `scripts/make_results_table.py` ordering, skipping and placeholders |
| `tests/test_batch.py` | Batch prediction used by the app's CSV mode |
| `tests/test_models_smoke.py` | Each committed model loads and predicts one sentence |

The smoke tests, and the LR tests in `test_batch.py`, skip when the weights are missing or are
LFS pointers. With `REQUIRE_MODELS=1` the smoke tests fail instead of skipping.

Continuous integration:

- `.github/workflows/tests.yml` runs on every push and pull request with lightweight dependencies
  and no LFS checkout, so model tests skip.
- `.github/workflows/models.yml` is started manually from the Actions tab. It checks out the LFS
  weights (about 571 MB per run, counted against the repository's LFS bandwidth), runs the tests
  with `REQUIRE_MODELS=1` using only the inference dependencies, then runs a short LR training.

## Deployment

The app runs locally with `streamlit run app/streamlit_app.py` or `python run_app.py`, after
`git lfs pull`. It has a single-text mode, a compare-all-models mode and a Batch CSV mode
(up to 1,000 rows, results downloadable as CSV).

A `Dockerfile` packages the app for local use:

```bash
docker build -t tweeteval-sentiment .
docker run --rm -p 8501:8501 tweeteval-sentiment
```

- The build fails with a list of files if any model file is still a Git LFS pointer
  (`scripts/check_lfs_pointers.py`).
- PyTorch is installed from the CPU-only index at the pinned version; the container does not use
  a GPU.
- `HEALTHCHECK` calls `/_stcore/health`, the health endpoint of Streamlit 1.51, which returns
  `ok`.
- The container runs as a non-root user.

No image is published to a registry.

## Appendix

### Dependencies

`requirements/inference.txt` pins the versions the models were saved with. Two pins need
explanation:

- `pyyaml==6.0.3` is not imported by project code. It is a declared dependency of
  `transformers` 4.57.3 (`pyyaml>=5.1`, also required by `huggingface_hub`), so the pin fixes a
  transitive version for reproducibility. It should stay.
- `dill==0.4.0` is not imported by project code either, but the committed
  `models/lstm/tokenizer.joblib` and `models/gru/tokenizer.joblib` cannot be loaded without it:
  they were pickled while `datasets` (which imports `dill`) was loaded, and contain a reference to
  `dill._dill._load_type`.

`requirements/train.txt` adds `datasets`, `evaluate`, `accelerate`, `pandas` and `tqdm`;
`requirements/dev.txt` adds `pytest` and `ruff`.

### Scripts

| Command | Purpose |
|---|---|
| `python -m src.training.train_lr` | Train LR; writes `models/lr/` and `results/lr.json` |
| `python -m src.training.train_lstm --map-emoticons` | Train BiLSTM (`--no-class-weight`, `--glove-path`, `--out-dir`, `--results-path`, `--subset` available) |
| `python -m src.training.train_gru --map-emoticons` | Train BiGRU (same options) |
| `python -m src.training.train_roberta` | Fine-tune RoBERTa from the task-neutral base (GPU needed in practice) |
| `python scripts/evaluate_models.py` | Evaluate committed models on the official splits; writes `results/eval/` |
| `python scripts/make_results_table.py` | Print the results table from `results/*.json` |
| `python scripts/check_lfs_pointers.py` | Fail if any file under `models/` is an LFS pointer |

Training scripts overwrite `models/<name>/` and `results/<name>.json` unless `--out-dir` and
`--results-path` are given.

### Results files

| File | Written by |
|---|---|
| `results/lr.json`, `lstm.json`, `gru.json` | Training scripts (validation and test blocks) |
| `results/roberta_legacy.json` | `scripts/evaluate_models.py --tag legacy` (`"comparable": false`) |
| `results/roberta_trainer_state_2025-12-05.json` | Hugging Face Trainer log of the legacy run |
| `results/eval/*.json` | `scripts/evaluate_models.py` (re-evaluation of the committed weights) |
