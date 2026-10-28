"""The committed RoBERTa model must keep the preprocessing it was trained with.

Its inference_config.json records preprocess_tweet with default arguments,
the same as the legacy fallback. These tests read only the JSON file, so
they run without torch or the LFS weights.
"""
from pathlib import Path

import pytest

from src.models.inference import CONFIG_FILENAME, load_inference_config, preprocess_for_model
from src.utils.preprocessing import preprocess_tweet

REPO = Path(__file__).resolve().parent.parent
ROBERTA_DIR = REPO / "models" / "roberta"

INPUTS = [
    "I love this so much :)",
    "worst day ever :(",
    "@user check this out http://t.co/abc123 #awesome",
    "The meeting is at 10:30 tomorrow.",
    "not good :( but ok <3 maybe :/",
    "Absolutely THRILLED with the new update \U0001F600",
    "I can't believe they cancelled the show again",
    "It's fine I guess",
    "",
    "Mixed CaSe With Trailing Spaces   ",
]


@pytest.fixture(scope="module")
def roberta_cfg():
    return load_inference_config(ROBERTA_DIR)


def test_roberta_config_file_is_used(roberta_cfg):
    # Without the file the legacy fallback would give the same answers, so check the source.
    assert Path(roberta_cfg["source"]) == ROBERTA_DIR / CONFIG_FILENAME


def test_roberta_labels(roberta_cfg):
    assert roberta_cfg["labels"] == ["negative", "neutral", "positive"]


@pytest.mark.parametrize("text", INPUTS)
def test_roberta_preprocessing_matches_default(roberta_cfg, text):
    assert preprocess_for_model(text, roberta_cfg) == preprocess_tweet(text)
