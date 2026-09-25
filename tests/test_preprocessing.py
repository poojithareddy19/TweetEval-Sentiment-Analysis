import re

import emoji
import pytest

from src.utils.preprocessing import preprocess_for_transformer, preprocess_tweet


def _legacy(text):
    """Verbatim copy of preprocess_tweet() as it was when the committed models were trained."""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = re.sub(r"http\S+", " <url> ", text)
    text = re.sub(r"@\w+", " <user> ", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    try:
        text = emoji.demojize(text)
    except Exception:
        pass
    text = re.sub(r"\s+", " ", text).strip()
    return text


SAMPLE_TWEETS = [
    "I love this so much :)",
    "worst day ever :(",
    "@user check this out http://t.co/abc123 #awesome",
    "  Multiple   spaces\tand\nnewlines  ",
    "HTTP://EXAMPLE.COM/Upper CASE url and HTTPS://x.y",
    "Meeting at 10:30, then 3:/4 done",
    "Emoji party \U0001F600 \U0001F44D ❤️",
    "#Hashtag #Another_tag @Mention @another_one",
    "not good :( but ok <3 maybe :/",
    "Mixed CaSe With Trailing Spaces   ",
    "",
    "just text",
    "ratio 1:2 and time 12:00:00",
    "İstanbul and straße and ﬁ ligature",
    "http://a.b :) http://c.d",
]


@pytest.mark.parametrize("tweet", SAMPLE_TWEETS)
def test_default_matches_legacy(tweet):
    assert preprocess_tweet(tweet) == _legacy(tweet)


@pytest.mark.parametrize("value", [None, 42, 3.5, ["list"], b"bytes"])
def test_non_string_returns_empty(value):
    assert preprocess_tweet(value) == ""
    assert preprocess_tweet(value, map_emoticons=True) == ""
    assert preprocess_for_transformer(value) == ""


def test_emoticon_mapping_opt_in():
    assert preprocess_tweet("I love this so much :)", map_emoticons=True) == "i love this so much smileface"
    assert preprocess_tweet("worst day ever :(", map_emoticons=True) == "worst day ever sadface"
    out = preprocess_tweet("good :-) and :D and =)", map_emoticons=True)
    assert out == "good smileface and smileface and smileface"
    assert preprocess_tweet("bad :-( and :'(", map_emoticons=True) == "bad sadface and sadface"
    assert preprocess_tweet("love <3 you", map_emoticons=True) == "love heartsymbol you"
    assert preprocess_tweet("hmm :/ not sure", map_emoticons=True) == "hmm skepticface not sure"


def test_emoticon_mapping_off_by_default():
    assert preprocess_tweet("worst day ever :(") == "worst day ever :("


def test_url_not_matched_as_skeptic_face():
    out = preprocess_tweet("see http://example.com/x now", map_emoticons=True)
    assert out == "see <url> now"
    assert "skepticface" not in out
    out = preprocess_tweet("HTTPS://Example.com :/", map_emoticons=True)
    assert out == "<url> skepticface"


def test_times_and_ratios_untouched():
    assert preprocess_tweet("meet at 10:30 ok", map_emoticons=True) == "meet at 10:30 ok"
    assert preprocess_tweet("score 3:/4 done", map_emoticons=True) == "score 3:/4 done"
    assert preprocess_tweet("x:)y", map_emoticons=True) == "x:)y"


def test_transformer_preprocessing_keeps_case_and_emoji():
    raw = "@Bob I LOVE this \U0001F600 http://t.co/xyz #Great"
    out = preprocess_for_transformer(raw)
    assert out == "@user I LOVE this \U0001F600 http #Great"
    assert preprocess_for_transformer("   padded   ") == "padded"
    assert preprocess_for_transformer("Hello :) <3") == "Hello :) <3"
