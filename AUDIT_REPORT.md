# Audit Report: TweetEval-Sentiment-Analysis

Branch `fix/doc-audit-gaps`, created from `main` at `d316d1f` (up to date with `origin/main` after `git fetch`).

## Baseline

Recorded on 2026-10-28, before any code or documentation change.

### Environment

- OS: Windows 11 (win32).
- Python 3.11.9 (official python.org installer, per-user), venv at `.venv`. Python 3.11 was not
  installed on the machine beforehand; only 3.15.1 and 3.12.16 were present.
- `pip install -r requirements/dev.txt -r requirements/train.txt`: succeeded with every pin
  resolved as written (transformers 4.57.3, torch 2.9.0, tensorflow 2.20.0, keras 3.12.0,
  scikit-learn 1.7.2, numpy 2.3.4, streamlit 1.51.0, pyyaml 6.0.3, datasets 4.4.1).

### Model files

`git lfs install && git lfs pull` completed. All LFS objects were already present locally.
`git lfs ls-files` lists 8 files, all checked out (`*`):

```
models/gru/best.keras (16 MB)
models/gru/model_final.keras (16 MB)
models/gru/tokenizer.joblib (1.8 MB)
models/lr/pipeline.joblib (2.6 MB)
models/lstm/best.keras (17 MB)
models/lstm/model_final.keras (17 MB)
models/lstm/tokenizer.joblib (1.8 MB)
models/roberta/model.safetensors (499 MB)
```

`find models -type f -size -1k` returns only `models/.gitkeep` (empty placeholder), so no pointer
files remain. The SHA-256 of every file under `models/` is recorded in
`outputs/audit/baseline_models.txt` (gitignored). For every LFS file the on-disk SHA-256 equals
the LFS object ID.

### Tests and lint

```
$ pytest -rs
collected 38 items
tests\test_io.py .....                                                   [ 13%]
tests\test_metrics.py ....                                               [ 23%]
tests\test_models_smoke.py ....                                          [ 34%]
tests\test_preprocessing.py .........................                    [100%]
============================= 38 passed in 45.40s =============================

$ ruff check .
All checks passed!
```

All 4 smoke tests in `tests/test_models_smoke.py` ran (none skipped).

### Baseline predictions

`outputs/audit/capture_predictions.py` mirrors `predict_proba()` in `app/streamlit_app.py`: each
input goes through `preprocess_for_model` with the model's inference config, then the same
per-model call (LR `predict_proba`, `keras_predict_proba` on `best.keras`, `RobertaWrapper` on CPU
with the app's default `max_length=128`). The empty input is skipped because the app rejects it.
Full probability vectors are in `outputs/audit/baseline_predictions.json` (gitignored).

A second run compared with the first gave a maximum absolute difference of 0 for all four
models, so the capture is deterministic on this machine.

Predicted label (highest probability) per model:

| # | Input | LR | BiLSTM | BiGRU | RoBERTa |
|---|---|---|---|---|---|
| 1 | `I love this so much :)` | positive (0.9870) | positive (0.9589) | positive (0.9834) | positive (0.9958) |
| 2 | `worst day ever :(` | negative (0.9231) | negative (0.9941) | negative (0.9916) | negative (0.9549) |
| 3 | `@user check this out http://t.co/abc123 #awesome` | positive (0.8591) | positive (0.7087) | positive (0.9265) | positive (0.9856) |
| 4 | `The meeting is at 10:30 tomorrow.` | neutral (0.6575) | neutral (0.7107) | neutral (0.8173) | neutral (0.9823) |
| 5 | `not good :( but ok <3 maybe :/` | negative (0.7576) | negative (0.8281) | negative (0.9185) | negative (0.8150) |
| 6 | `Absolutely THRILLED with the new update \U0001f600` | positive (0.6144) | positive (0.9132) | positive (0.9301) | positive (0.9943) |
| 7 | `I can't believe they cancelled the show again` | negative (0.9462) | negative (0.9540) | negative (0.8930) | negative (0.9463) |
| 8 | `It's fine I guess` | neutral (0.3912) | neutral (0.4176) | positive (0.4827) | positive (0.6268) |
| 9 | `Mixed CaSe With Trailing Spaces   ` | neutral (0.7640) | neutral (0.7946) | neutral (0.8666) | neutral (0.9752) |

### Notes

- `docs/DOCUMENTATION.md` does not exist yet (no `docs/` directory).
- `models/roberta/` has no `inference_config.json`, so it loads `LEGACY_CONFIG` (`max_len` 80),
  but neither the app nor `RobertaWrapper` reads `max_len`; RoBERTa truncates at 128 tokens.

## Phase 1: Low-risk correctness and documentation fixes

### Commits

```
7f93d08 Correct README claims, project tree and add test instructions
2af1b6f Document why pyyaml is pinned
a6a8c14 Add inference config for legacy RoBERTa model
8886f17 Use sentiment label names in RoBERTa config
```

After each commit `pytest -q` and `ruff check .` passed (38 tests for the first two, 50 after
`tests/test_inference_config.py` was added).

### 1.1 README

- LR described as a classical baseline, not deep learning.
- "Fast Inference" replaced by two facts verified in `app/streamlit_app.py`: every loader is
  wrapped in `@st.cache_resource`, and `RobertaWrapper` (hence torch and transformers) is only
  imported inside `load_roberta`.
- "after `git lfs pull`" added to the Features line and to the matching line under Notes.
- Test-split sentence reworded; `results/lr.json`, `lstm.json` and `gru.json` each contain a
  `test` block, the RoBERTa Trainer state does not.
- Project tree renamed to `TweetEval-Sentiment-Analysis/` with `.github/workflows/`, `results/`,
  `scripts/`, `tests/`, `requirements/dev.txt`, `pyproject.toml` and `run_app.py`.
- "Running tests" section and `python run_app.py` added. Results numbers unchanged.

### 1.2 RoBERTa inference config

Evidence from git history that the legacy config is correct:

- `model.safetensors` was added in `2f33e2b` (2025-12-06) as blob `be68ee0`; `HEAD` has the same
  blob at `models/roberta/model.safetensors` (only renamed since).
- `src/training/train_bert.py` at `59dc788` (2025-12-04) and `7db6bd0` (2025-12-06) uses
  `MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"`, `MAX_LEN = 128` and
  `texts = [preprocess_tweet(t) for t in examples["text"]]` (default arguments).
- `src/utils/preprocessing.py` at `59dc788` has the single-argument `preprocess_tweet`, identical
  to `_legacy` in `tests/test_preprocessing.py`.

`models/roberta/inference_config.json` written with `save_json` using the planned content.
`tests/test_inference_config.py` (12 tests) checks that the file itself is loaded (not the
fallback), the labels, and that `preprocess_for_model` equals `preprocess_tweet` on the Phase 0
inputs. It imports neither torch, tensorflow nor transformers (checked with `python -X importtime`).

### 1.3 RoBERTa label names

`grep -rnE "LABEL_[0-9]|id2label|label2id|config\.labels|pipeline\("` outside `.venv`, `.git` and
`outputs` matched only `models/roberta/config.json` and `src/training/train_roberta.py` (which
already writes `negative/neutral/positive`). The app takes labels from the inference config.
`id2label` and `label2id` changed; `AutoConfig.from_pretrained("models/roberta")` now returns
`{0: 'negative', 1: 'neutral', 2: 'positive'}`.

### 1.4 pyyaml

`pip show pyyaml`: `Required-by: accelerate, datasets, huggingface_hub, transformers`;
transformers 4.57.3 declares `pyyaml>=5.1`. No project code imports yaml. Comment added above the
pin.

### 1.5

`docs/DOCUMENTATION.md` does not exist yet; deferred to Phase 6.

### Behaviour check

```
$ python outputs/audit/capture_predictions.py outputs/audit/after_phase1.json --compare outputs/audit/baseline_predictions.json
lr       allclose(atol=1e-06) = True  max_abs_diff = 0
lstm     allclose(atol=1e-06) = True  max_abs_diff = 0
gru      allclose(atol=1e-06) = True  max_abs_diff = 0
roberta  allclose(atol=1e-06) = True  max_abs_diff = 0
```

RoBERTa's `config_source` is now `models/roberta/inference_config.json`. SHA-256 of every
non-JSON file under `models/` matches `outputs/audit/baseline_models.txt`.

## Phase 2: Reproducibility and evaluation of the committed models

### Commits

```
097e8f3 Add batch prediction helper for Keras models
809d115 Add script to evaluate committed models on TweetEval
dccbaf2 Add re-evaluation results for LR, BiLSTM and BiGRU
1a54b54 Add tests for the results table script
43c5f75 Add evaluation results for legacy RoBERTa model
```

`pytest -q` and `ruff check .` passed after each commit (57 tests after `test_results_table.py`).

### 2.1 Evaluation script

- `src/models/inference.py`: new `keras_sequences()` (post padding and truncation to `max_len`)
  shared by `keras_predict_proba()` and the new `keras_predict_proba_batch()`, which calls
  `model.predict(..., batch_size=64)` like `src/training/rnn_common.py`. Baseline predictions
  re-captured after this refactor: max_abs_diff 0 for all four models.
- `scripts/evaluate_models.py`: options `--models`, `--splits`, `--out-dir` (default
  `results/eval`), `--subset`, `--device`, `--batch-size`, `--tag`, `--extra-json`. Output JSON
  has the training scripts' `validation` and `test` blocks plus `model_path`, `config_source`,
  `evaluated_with` and `subset`.
- `python scripts/make_results_table.py` output is unchanged with `results/eval/` present.

### 2.2 Reproduction check

`python scripts/evaluate_models.py --models lr lstm gru` (Keras models evaluated from
`best.keras`, the file the app loads first). Full comparison in
`outputs/audit/reproduction_check.txt`; test split:

| Model | Split | Metric | Committed | Re-evaluated | Abs diff | Within 0.001 |
| LR | test | accuracy | 0.5922 | 0.5922 | 0.00e+00 | yes |
| LR | test | f1_macro | 0.5887 | 0.5887 | 0.00e+00 | yes |
| LR | test | recall_macro | 0.5987 | 0.5987 | 0.00e+00 | yes |
| BiLSTM | test | accuracy | 0.5785 | 0.5785 | 0.00e+00 | yes |
| BiLSTM | test | f1_macro | 0.5726 | 0.5726 | 0.00e+00 | yes |
| BiLSTM | test | recall_macro | 0.5841 | 0.5841 | 0.00e+00 | yes |
| BiGRU | test | accuracy | 0.5693 | 0.5693 | 0.00e+00 | yes |
| BiGRU | test | f1_macro | 0.5641 | 0.5641 | 0.00e+00 | yes |
| BiGRU | test | recall_macro | 0.5799 | 0.5799 | 0.00e+00 | yes |

Every metric on both splits differs by exactly 0, and the validation and test confusion
matrices are identical for all three models (not only LR). Files evaluated:
`models/lr/pipeline.joblib`, `models/lstm/best.keras`, `models/gru/best.keras`.

### 2.3 Legacy RoBERTa

`python scripts/evaluate_models.py --models roberta --device cpu` on a CPU-only torch build
(2.9.0+cpu): validation 98.5 s, test 494.7 s. Output `results/eval/roberta.json`.

| Split | Accuracy | Macro F1 | Macro Recall |
|---|---|---|---|
| validation | 0.7890 | 0.7806 | 0.7824 |
| test | 0.7142 | 0.7131 | 0.7179 |

Validation accuracy (0.789) and macro F1 (0.7805747389723933) are identical to the epoch 1 entry
and `best_metric` in `results/roberta_trainer_state_2025-12-05.json`, whose
`best_model_checkpoint` is `checkpoint-5702`; `2f33e2b` shows that checkpoint and the shipped
`model.safetensors` are the same blob. This confirms both the shipped weights and the
preprocessing recorded in 1.2. These numbers are not comparable with LR, BiLSTM and BiGRU
because the base model was already fine-tuned on TweetEval sentiment.

Approved and applied at Checkpoint 2:

```
197ecad Show legacy RoBERTa results last in the results table
e692f84 Add test-split results for legacy RoBERTa model
d86c813 Add legacy RoBERTa row to README results
```

- `results/roberta_legacy.json` written only by the script:
  `python scripts/evaluate_models.py --models roberta --device cpu --out-dir results --tag legacy
  --extra-json '{"comparable": false, "base_model": "cardiffnlp/twitter-roberta-base-sentiment"}'`
  (validation 93.3 s, test 453.1 s). Its validation and test blocks are identical to
  `results/eval/roberta.json`.
- `scripts/make_results_table.py`: display name `roberta_legacy` added, last in `ORDER`.
- README results table replaced with the output of `python scripts/make_results_table.py`; text
  now says no comparable RoBERTa row exists and points to the legacy row. The explanation of why
  it is not comparable is unchanged.

### 2.4 Results table tests

`tests/test_results_table.py` (7 tests, standard library only) loads the script with
`importlib.util.spec_from_file_location` and covers ORDER, unknown keys appended after known
ones, files without the requested split (Trainer state shape) skipped, the validation split,
missing metrics shown as `-`, and the placeholder row for an empty directory or one with no
usable files.

## Phase 3: Make CI exercise the real models

### Commits

```
1d12258 Fail smoke tests instead of skipping when REQUIRE_MODELS=1
7269c6e Add manual workflow that tests the real models
```

`pytest -q` (57 passed) and `ruff check .` passed after each commit.

### 3.1 Strict mode

`tests/test_models_smoke.py` routes every skip (missing file, LFS pointer, no `.keras` file,
`importorskip` for tensorflow, torch and transformers) through `_skip` / `_importorskip`, which
call `pytest.fail` when `REQUIRE_MODELS=1`. Default behaviour is unchanged.

Verified on a clone without LFS (`GIT_LFS_SKIP_SMUDGE=1 git clone`, pointer files only):

```
$ pytest -rs -q tests/test_models_smoke.py
SKIPPED [1] tests\test_models_smoke.py:26: models\lr\pipeline.joblib is a Git LFS pointer, run `git lfs pull`
SKIPPED [1] tests\test_models_smoke.py:26: models\lstm\tokenizer.joblib is a Git LFS pointer, run `git lfs pull`
SKIPPED [1] tests\test_models_smoke.py:26: models\gru\tokenizer.joblib is a Git LFS pointer, run `git lfs pull`
SKIPPED [1] tests\test_models_smoke.py:26: models\roberta\model.safetensors is a Git LFS pointer, run `git lfs pull`
4 skipped

$ REQUIRE_MODELS=1 pytest -q tests/test_models_smoke.py
FAILED tests/test_models_smoke.py::test_lr_predicts - Failed: REQUIRE_MODELS=...
FAILED tests/test_models_smoke.py::test_keras_rnn_predicts[lstm] - Failed: RE...
FAILED tests/test_models_smoke.py::test_keras_rnn_predicts[gru] - Failed: REQ...
FAILED tests/test_models_smoke.py::test_roberta_predicts - Failed: REQUIRE_MO...
4 failed
```

### 3.2 Manual model workflow

`.github/workflows/models.yml`: `workflow_dispatch` only, weekly `schedule` commented out with a
bandwidth note. The LFS objects total 570,695,466 bytes (about 571 MB, 544 MiB), measured with
`git lfs ls-files` and `stat`; the plan's estimate was about 535 MB. Steps: checkout with
`lfs: true`, Python 3.11, install `requirements/dev.txt` and `requirements/train.txt`, pointer
check with `src.utils.io.is_lfs_pointer`, `REQUIRE_MODELS=1 pytest -rs`, and
`python -m src.training.train_lr --subset 300 --out-dir /tmp/lr_smoke --results-path /tmp/lr_smoke.json`.
`tests.yml` is unchanged. The YAML parses (`yaml.safe_load`) with `workflow_dispatch` as the only
trigger.

Local equivalent (full log in `outputs/audit/phase3_local_run.txt`; the training output went to
the session scratch directory instead of `/tmp`):

```
No LFS pointers under models/                       (exit 0)
REQUIRE_MODELS=1 pytest -rs                         57 passed
train_lr --subset 300                               36 fits, results and pipeline saved (exit 0)
```

`git status` afterwards showed no change under `models/` or `results/`. The workflow itself can
only be verified on GitHub after the branch is pushed.

## Phase 4: Feature gaps

### Commits so far

```
11fe58a Add batch prediction module for many texts
0178334 Add batch CSV prediction mode to the app
d03f3f5 Add LFS pointer check script and use it in the models workflow
```

`pytest -q` (66 passed) and `ruff check .` passed after each commit.

### 4.1 Batch CSV prediction

- `src/models/batch.py`: `predict_proba_batch(kind, model, raw_texts, cfg)` (numpy only, applies
  `preprocess_for_model`, reuses `keras_predict_proba_batch`) and `prediction_columns()` building
  `prediction`, `confidence` and `prob_<label>` columns. `MAX_BATCH_ROWS = 1000`.
- `tests/test_batch.py` (9 tests): LR batch equals single predictions (skips on pointer), NaN and
  None rows, empty list for every model kind, non-string cells reach the model as `""`, unknown
  kind, column building.
- `app/streamlit_app.py`: sidebar radio "Mode" (Single text, Batch CSV). Batch mode runs before the
  single-text UI and ends with `st.stop()`; "Compare all models" is only shown in single-text mode.

Checked in the browser against a headless server (`streamlit run ... --server.headless true`):

- Single text, LR, "worst day ever :(": NEGATIVE 0.9231 / 0.0519 / 0.0250 (baseline).
- Compare all models, same text: LR 0.923, LSTM 0.994, GRU 0.992, RoBERTa 0.955 (baseline, 3 dp).
- Batch CSV, LR, 5 rows with one empty cell: confidences 0.987, 0.9231, 0.5502 (empty, neutral),
  0.6575, 0.8591 (baseline for the four non-empty rows). Download button shown.
- Batch CSV, 1,200 rows: warning "The file has 1200 rows. Only the first 1000 are used."; GRU
  batch prediction on the 1,000 rows succeeded.
- Server log: no errors; only TensorFlow's own `tf.reset_default_graph` deprecation notice.

### 4.2 Container (in progress)

- Health endpoint: Streamlit 1.51 `web/server/server.py` defines `NEW_HEALTH_ENDPOINT =
  "_stcore/health"`; `/healthz` still matches but `routes.py` emits a deprecation notice for it.
- `scripts/check_lfs_pointers.py` wraps `src.utils.io.is_lfs_pointer`; exit 0 on this checkout,
  exit 1 listing all 8 files on a `GIT_LFS_SKIP_SMUDGE=1` clone. `models.yml` now calls it.
- `Dockerfile` (`python:3.11-slim`, torch from the PyTorch CPU index with the version read from
  `requirements/inference.txt`, then `pip install -r requirements/inference.txt`, pointer check,
  non-root `appuser`, `HEALTHCHECK` on `/_stcore/health` via `urllib`) and `.dockerignore`.
- Build: `torch-2.9.0+cpu` (184.5 MB wheel), no `nvidia-*` packages, pointer check
  "No LFS pointers under /app/models". Image 6.24 GB (7.35 GB before removing a `chown -R` layer
  that duplicated `/app`); site-packages 3.6 GB (tensorflow 1.8 GB, torch 728 MB), models 550 MB.
- Run: `GET /_stcore/health -> 200 ok`, `GET / -> 200`, Docker health status `healthy`, user
  `appuser`. In the container, LR and RoBERTa predictions for "worst day ever :(" match the
  baseline (0.9231 / 0.0519 / 0.025 and 0.9549 / 0.0407 / 0.0044).

**Finding: LSTM and GRU cannot be loaded with `requirements/inference.txt` alone.**
`load_keras_model` fails with `ModuleNotFoundError: No module named 'dill'`. Both
`models/*/tokenizer.joblib` contain a `dill._dill._load_type('int')` global (pickletools offset
549184), written because `datasets` (which imports dill) was loaded when the tokenizers were
pickled. `dill` 0.4.0 is installed locally only as a dependency of `datasets`, `evaluate` and
`multiprocess` from `requirements/train.txt`, so local tests and the models workflow do not
catch it. In a throwaway container with `pip install dill==0.4.0` added, LSTM (0.9941) and GRU
(0.9916) match the baseline.

### 4.2 Fix and completion (approved: pin dill==0.4.0)

```
83cec14 Add dill to inference requirements for the Keras tokenizers
1f46cd4 Run model tests with inference dependencies only
992fbe4 Add Dockerfile for running the app locally
98e0388 Document Docker, batch CSV mode and strict model tests in README
```

- `requirements/inference.txt`: `dill==0.4.0` with a comment explaining the tokenizer pickles.
- Guard: `models.yml` now installs only `requirements/dev.txt` (inference plus pytest and ruff)
  before the pointer check and `REQUIRE_MODELS=1 pytest -rs`, and installs
  `requirements/train.txt` afterwards for the training smoke test. A model that needs a
  training-only package now fails the test step.
- Local equivalent of the guard: in the rebuilt image (inference dependencies only; `datasets`,
  `evaluate` and `multiprocess` absent, `dill` 0.4.0 present) plus `pytest==8.4.2`,
  `REQUIRE_MODELS=1 python -m pytest -rs` gave 66 passed.
- Rebuilt image (6.24 GB): `GET /_stcore/health -> 200 ok`, `GET / -> 200`, health `healthy`,
  user `appuser`. Predictions for "worst day ever :(" inside the container, rounded to 4 dp:
  LR 0.9231 / 0.0519 / 0.025, LSTM 0.9941 / 0.0055 / 0.0004, GRU 0.9916 / 0.0073 / 0.0011,
  RoBERTa 0.9549 / 0.0407 / 0.0044, all equal to the baseline.
- A CRLF copy of the Dockerfile (as a Windows checkout with `core.autocrlf=true` produces) builds
  the same image with a clean `HEALTHCHECK` command.
- No image was pushed. Local images left behind: `tweeteval-sentiment:local`,
  `tweeteval-sentiment:crlf-check` (same layers) and the untagged first build (7.35 GB, with the
  removed `chown -R` layer); deleting them needs user approval.
- README: Docker and Batch predictions subsections, Batch CSV feature line, `REQUIRE_MODELS=1`
  note, `Dockerfile`, `check_lfs_pointers.py`, `evaluate_models.py` and `models.yml` in the tree.
  `docs/DOCUMENTATION.md` does not exist, so no docs change.

### Behaviour check after Phase 4

`capture_predictions.py ... --compare baseline_predictions.json`: max_abs_diff 0 for LR, LSTM,
GRU and RoBERTa. SHA-256 of every non-JSON file under `models/` unchanged.

## Phase 5: Model experiments

### 5.1 Class-weight ablation (BiLSTM, `--map-emoticons`)

Commit `af4d940 Add --no-class-weight option to RNN training`: `--no-class-weight` for
`train_lstm` and `train_gru`, `train_rnn(use_class_weight=True)` by default, and
`use_class_weight` recorded in the results JSON (`class_weight` is `null` when disabled).
Default behaviour unchanged; `pytest -q` 66 passed, `ruff check .` clean. Subset smoke runs
(`--subset 300 --epochs 1`) recorded `True` with balanced weights and `False` / `null`.

Timing: one full epoch 106 s (713 steps, 141 ms/step) on CPU. Runs (approved at Checkpoint 4,
both written only to `outputs/experiments/`):

```
python -m src.training.train_lstm --map-emoticons --out-dir outputs/experiments/lstm_cw --results-path outputs/experiments/lstm_cw.json                      # 451 s
python -m src.training.train_lstm --map-emoticons --no-class-weight --out-dir outputs/experiments/lstm_nocw --results-path outputs/experiments/lstm_nocw.json  # 352 s
```

Validation loss per epoch (EarlyStopping, patience 2, best weights restored): with class
weights 0.8619, 0.8598, 0.9506, 1.1126 (best epoch 2); without 0.7477, 0.7728, 0.9236 (best
epoch 1).

Tables generated by `python outputs/experiments/compare_class_weight.py` from the JSON files
(also saved as `outputs/experiments/class_weight_comparison.md`). "Neutral predicted negative"
is confusion matrix cell [1][0] over the neutral row total.


test split

| Run | use_class_weight | Epochs run | Accuracy | Macro F1 | Macro recall | Recall neg / neu / pos | Neutral predicted negative | Negative predicted neutral |
|---|---|---|---|---|---|---|---|---|
| Committed BiLSTM (results/lstm.json) | True | - | 0.5785 | 0.5726 | 0.5841 | 0.7729 / 0.4800 / 0.4994 | 2475 of 5937 (41.7%) | 733 of 3972 (18.5%) |
| BiLSTM, class weights (rerun) | True | 4 | 0.5773 | 0.5700 | 0.5810 | 0.7822 / 0.4782 / 0.4825 | 2527 of 5937 (42.6%) | 707 of 3972 (17.8%) |
| BiLSTM, no class weights | False | 3 | 0.6193 | 0.5979 | 0.5841 | 0.5624 / 0.7150 / 0.4749 | 1209 of 5937 (20.4%) | 1567 of 3972 (39.5%) |

validation split

| Run | use_class_weight | Epochs run | Accuracy | Macro F1 | Macro recall | Recall neg / neu / pos | Neutral predicted negative | Negative predicted neutral |
|---|---|---|---|---|---|---|---|---|
| Committed BiLSTM (results/lstm.json) | True | - | 0.6145 | 0.6053 | 0.6579 | 0.8205 / 0.5719 / 0.5812 | 264 of 869 (30.4%) | 36 of 312 (11.5%) |
| BiLSTM, class weights (rerun) | True | 4 | 0.6110 | 0.6020 | 0.6570 | 0.8269 / 0.5420 / 0.6020 | 280 of 869 (32.2%) | 34 of 312 (10.9%) |
| BiLSTM, no class weights | False | 3 | 0.6620 | 0.6447 | 0.6613 | 0.6635 / 0.7123 / 0.6081 | 159 of 869 (18.3%) | 78 of 312 (25.0%) |

Findings (one run per setting, so no estimate of run-to-run variance):

- The rerun with class weights is close to the committed model (test macro recall 0.5810 vs
  0.5841), so the comparison is like for like.
- Without class weights the neutral-to-negative cell on test falls from 2527 (42.6% of neutral)
  to 1209 (20.4%), supporting the hypothesis that the balanced weights push neutral tweets
  towards negative.
- The error moves rather than disappears: negative-to-neutral rises from 707 (17.8%) to 1567
  (39.5%), and negative recall falls from 0.7822 to 0.5624.
- Test macro recall, the official TweetEval metric, is unchanged (0.5810 with, 0.5841 without,
  0.5841 committed). Test accuracy rises from 0.5773 to 0.6193 and macro F1 from 0.5700 to 0.5979.

No promotion proposed: the run without class weights is not better on the official metric.

### 5.2 and 5.3

Not run, by decision at Checkpoint 5 ("Skip to Phase 6"):

- 5.2 GloVe: needs the official Twitter GloVe archive (one zip of about 1.4 GB); not downloaded.
- 5.3 RoBERTa from the task-neutral base: `nvidia-smi` shows an NVIDIA GeForce GTX 1650 (4096 MiB),
  but the installed torch is `2.9.0+cpu` (`torch.cuda.is_available()` is False). A GPU run would
  need a CUDA torch build (several GB) and 4 GB of VRAM is tight for RoBERTa-base; the plan rules
  out a CPU run.

No model was promoted; nothing under `models/` changed in Phase 5.

## Phase 6: Final audit

Command output is saved under `outputs/audit/phase6_*.txt` (gitignored).

### 6.1 Automated checks

1. `pytest -rs` with full dependencies: `66 passed`, no skips (smoke tests ran).
2. `REQUIRE_MODELS=1 pytest -rs`: `66 passed`.
3. CI-equivalent run. A Windows venv in the session scratch directory failed to install
   scikit-learn because of the 260-character path limit (`OSError: [Errno 2]` on a joblib test
   file), an environment issue unrelated to the project, so the check ran on Linux like CI: a fresh
   `python:3.11-slim` container (Python 3.11.18) with only `emoji==2.15.0 numpy==2.3.4
   scikit-learn==1.7.2 pytest==8.4.2 ruff==0.13.3`, on a `GIT_LFS_SKIP_SMUDGE=1` clone of the
   branch at `af4d940` (`models/lr/pipeline.joblib` starts with `version https://git-lfs...`):

   ```
   SKIPPED [1] tests/test_batch.py:40: models/lr/pipeline.joblib missing or a Git LFS pointer
   SKIPPED [1] tests/test_batch.py:51: models/lr/pipeline.joblib missing or a Git LFS pointer
   SKIPPED [1] tests/test_models_smoke.py:26: models/lr/pipeline.joblib is a Git LFS pointer, run `git lfs pull`
   SKIPPED [2] tests/test_models_smoke.py:35: could not import 'tensorflow': No module named 'tensorflow'
   SKIPPED [1] tests/test_models_smoke.py:35: could not import 'torch': No module named 'torch'
   60 passed, 6 skipped
   ruff check .  ->  All checks passed!
   ```

   Only model-dependent tests skip; the new tests (`test_inference_config.py`,
   `test_results_table.py`, the non-model parts of `test_batch.py`) run without heavy imports.
4. `ruff check .`: `All checks passed!`
5. `git diff main -- tests/test_preprocessing.py`: empty (0 lines); `_legacy` unchanged.

### 6.2 Behaviour unchanged

6. `python outputs/audit/capture_predictions.py outputs/audit/final_predictions.json --compare
   outputs/audit/baseline_predictions.json`:

   ```
   lr       allclose(atol=1e-06) = True  max_abs_diff = 0
   lstm     allclose(atol=1e-06) = True  max_abs_diff = 0
   gru      allclose(atol=1e-06) = True  max_abs_diff = 0
   roberta  allclose(atol=1e-06) = True  max_abs_diff = 0
   ```

   No model was replaced, so all four are compared.
7. `streamlit run app/streamlit_app.py --server.headless true --server.port 8504`:
   `GET /_stcore/health -> 200 ok`, `GET / -> 200`, no errors in the log; the process was then
   stopped and port 8504 confirmed closed. (The app was also exercised in the browser in Phase 4.)

### 6.3 Repository hygiene

8. `git diff --stat main -- models/`:

   ```
   models/roberta/config.json           | 12 ++++++------
   models/roberta/inference_config.json | 15 +++++++++++++++
   2 files changed, 21 insertions(+), 6 deletions(-)
   ```

   Both approved at Checkpoint 1. SHA-256 of all 11 non-JSON files under `models/` (weights,
   tokenizers, vocab, merges) is identical to `outputs/audit/baseline_models.txt`.
9. `git lfs ls-files` still lists all 8 binaries (3 under `gru/`, 1 `lr/`, 3 `lstm/`,
   `roberta/model.safetensors`). The 15 files added on the branch are all text; the largest is
   `scripts/evaluate_models.py` at 5,044 bytes (limit 1 MB).
10. Em dash scan over `git diff main --name-only` plus this report: no matches (`grep` printed
    nothing; `xargs` exit 123 means every `grep` returned 1). Commit messages: no matches (exit 1).
11. Attribution scan `git log main..HEAD --format=%B | grep -inE "co-authored-by|generated
    with|anthropic|claude"`: no matches (exit 1).
12. `python outputs/audit/check_readme_numbers.py`: all 12 numbers in the README results table
    match `results/*.json` at 4 decimal places, and the output of
    `python scripts/make_results_table.py` appears verbatim in the README. The prose figures
    (0.789, 0.781, validation loss 0.51 to 0.98) come from
    `results/roberta_trainer_state_2025-12-05.json` (`log_history`: 0.789, 0.78057, 0.5146 to
    0.9767). `docs/DOCUMENTATION.md` does not exist, so it was skipped.

### 6.4 Gap status

| # | Gap (from docs) | Fix applied | Files changed | Evidence | Status |
|---|---|---|---|---|---|
| 1 | No comparable RoBERTa result | Legacy model evaluated on the official splits and shown last in the table, labelled not comparable | `scripts/evaluate_models.py`, `results/eval/roberta.json`, `results/roberta_legacy.json`, `scripts/make_results_table.py`, `README.md` | Test 0.7142 / 0.7131 / 0.7179; validation macro F1 equals Trainer `best_metric` exactly | Partially fixed: a comparable retrain from the task-neutral base (5.3) is deferred because torch here is CPU-only, the GPU has 4 GB, and a CPU run is excluded |
| 2 | RoBERTa has no inference config; generic labels | Explicit `inference_config.json` with legacy preprocessing; `id2label` / `label2id` renamed | `models/roberta/inference_config.json`, `models/roberta/config.json`, `tests/test_inference_config.py` | Git history (`59dc788`, `2f33e2b`); 12 new tests; `AutoConfig` returns `negative/neutral/positive`; predictions max_abs_diff 0 | Fixed |
| 3 | Validation-to-test drop, modest scores | Reproduced exactly; class-weight ablation run | `results/eval/*.json`, `src/training/rnn_common.py` | Re-evaluation diff 0 for LR, BiLSTM, BiGRU; ablation table in Phase 5 | Deferred: confirmed as real, no experiment improved the official metric, GloVe (5.2) skipped |
| 4 | RNN neutral-to-negative confusion | `--no-class-weight` ablation | `src/training/rnn_common.py`, `train_lstm.py`, `train_gru.py` | Test neutral-to-negative 42.6% with weights, 20.4% without; negative-to-neutral 17.8% to 39.5%; macro recall 0.5810 vs 0.5841 | Deferred: cause confirmed, but removing the weights moves the error and does not improve macro recall, so no promotion |
| 5 | No RNN hyperparameter search; GloVe unused | None | none | 5.2 not run | Deferred: GloVe needs a 1.4 GB download, skipped at Checkpoint 5; no search run |
| 6 | Single-text interface only | Batch CSV mode with download, 1,000-row cap | `src/models/batch.py`, `app/streamlit_app.py`, `tests/test_batch.py`, `README.md` | 9 tests; browser check: batch confidences equal baseline, cap warning, GRU batch on 1,000 rows | Fixed |
| 7 | Heavy runtime dependencies | Container with CPU-only torch; missing `dill` added | `Dockerfile`, `.dockerignore`, `requirements/inference.txt`, `README.md` | Image 6.24 GB, no `nvidia-*` packages; 66 tests pass in the image with inference deps only | Partially fixed: packaged and documented, footprint itself unchanged (TensorFlow 1.8 GB, torch 0.7 GB) |
| 8 | CI does not exercise real models or training | `REQUIRE_MODELS=1` strict mode; manual `models.yml` (inference deps only for tests, then LR training smoke) | `tests/test_models_smoke.py`, `.github/workflows/models.yml`, `scripts/check_lfs_pointers.py` | No-LFS clone: 4 skipped by default, 4 failed with `REQUIRE_MODELS=1`; all workflow steps pass locally | Partially fixed: the workflow has not run on GitHub yet (needs a push) |
| 9 | Training scope limited to TweetEval | None | none | Documented only, per plan | Deferred: out of scope for this branch |
| 10 | pyyaml pin and unused `split.py` | pyyaml pin explained; `split.py` removed after approval (follow-up) | `requirements/inference.txt`, `src/models/split.py` | `pip show pyyaml`: required by transformers (`pyyaml>=5.1`), accelerate, datasets, huggingface_hub; no imports of `split.py` before removal; tests pass after | Fixed |
| 11 | README discrepancies (5 items) | All 5 corrected, plus tests, Docker and batch sections | `README.md` | Phase 1 diff; check 12 | Fixed |
| 12 | `make_results_table.py` untested | 7 tests | `tests/test_results_table.py` | Pass in full and lightweight environments | Fixed |
| 13 | No deployment configuration | Dockerfile with LFS pointer check and health check | `Dockerfile`, `.dockerignore`, `scripts/check_lfs_pointers.py`, `README.md` | Build OK; `/_stcore/health` 200 ok, healthy, all 4 models equal baseline in the container; nothing pushed | Fixed (local only) |

Additional finding, fixed: `requirements/inference.txt` could not load the LSTM and GRU models
because the tokenizer pickles need `dill` (Phase 4). Fixed by pinning `dill==0.4.0`, guarded by
the inference-only test step in `models.yml`.

## Follow-up after merge

After PR 3 was merged, on branch `docs/documentation-and-cleanup` from `main` at `c5a6984`:

- `src/models/split.py` removed (approved). `grep` found no imports or references outside this
  report; `pytest -q` 66 passed and `ruff check .` clean afterwards.
- `docs/DOCUMENTATION.md` created (approved; plan item 1.5): overview, data, models, results,
  limitations, future enhancements, testing, deployment and an appendix that explains the
  `pyyaml` and `dill` pins. Split sizes and label counts come from `load_dataset`.
- `python outputs/audit/check_readme_numbers.py` (now checks results tables against the test or
  validation split and the per-class table against recall and confusion matrix cell [1][0]):
  README test table and the docs test, validation and per-class tables all match `results/*.json`;
  `make_results_table.py` output (test and validation) appears verbatim in the docs. `ALL MATCH`.
