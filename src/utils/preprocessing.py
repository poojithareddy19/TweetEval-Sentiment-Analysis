"""Text preprocessing shared by training scripts and the Streamlit app.

preprocess_tweet() is the function the committed LR, LSTM and GRU models were
trained with. Its default behaviour must stay byte-for-byte identical, so any
new behaviour is opt-in through keyword arguments and recorded per model in
models/<name>/inference_config.json.
"""
import re
from typing import Optional

import emoji

# Case-insensitive so that lowercasing can happen after URL replacement without
# changing the legacy output ("HTTP://..." was previously lowercased first).
_URL_RE = re.compile(r"http\S+", re.IGNORECASE)
_USER_RE = re.compile(r"@\w+")
_HASHTAG_RE = re.compile(r"#(\w+)")
_WS_RE = re.compile(r"\s+")

# Common text emoticons mapped to single word tokens. The Keras Tokenizer's
# default filters strip punctuation, so without this "not good :(" becomes
# ["not", "good"] and the sentiment cue is lost.
EMOTICON_TOKENS = {
    "smileface": [":)", ":-)", ":D", ":-D", "=)", "=D", ":]", ":-]", "(:", "(-:"],
    "sadface": [":(", ":-(", ":'(", ":'-(", "=(", ":[", ":-[", "):", ")-:"],
    "heartsymbol": ["<3"],
    "skepticface": [":/", ":-/", ":\\", ":-\\"],
}

def _build_emoticon_regex():
    # Longest first so ":-)" is not consumed as ":" + "-)" by a shorter pattern.
    pairs = [(tok, pat) for tok, pats in EMOTICON_TOKENS.items() for pat in pats]
    pairs.sort(key=lambda p: len(p[1]), reverse=True)
    alternation = "|".join(re.escape(pat) for _, pat in pairs)
    # Whitespace lookarounds: only match emoticons that stand alone as a token,
    # so times like "10:30" or ratios like "3:/4" are left untouched.
    return re.compile(r"(?<!\S)(?:%s)(?!\S)" % alternation), dict((pat, tok) for tok, pat in pairs)

_EMOTICON_RE, _EMOTICON_LOOKUP = _build_emoticon_regex()

def map_emoticons_to_tokens(text: str) -> str:
    """Replace standalone text emoticons with word tokens (smileface, sadface, ...)."""
    return _EMOTICON_RE.sub(lambda m: " %s " % _EMOTICON_LOOKUP[m.group(0)], text)

def preprocess_tweet(text: Optional[str], map_emoticons: bool = False) -> str:
    """Preprocess a tweet for the LR, LSTM and GRU models.

    With default arguments this produces exactly the output the committed
    models were trained on: strip, lowercase, URLs to <url>, @mentions to
    <user>, hashtags to bare words, emoji demojized, whitespace collapsed.

    map_emoticons: opt-in. Replace text emoticons such as ":)" and ":(" with
    word tokens before lowercasing. Applied after URL replacement so that ":/"
    inside "http://" is never matched.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = _URL_RE.sub(" <url> ", text)
    if map_emoticons:
        text = map_emoticons_to_tokens(text)
    text = text.lower()
    text = _USER_RE.sub(" <user> ", text)
    text = _HASHTAG_RE.sub(r"\1", text)
    try:
        text = emoji.demojize(text)
    except Exception:
        pass
    text = _WS_RE.sub(" ", text).strip()
    return text

def preprocess_for_transformer(text: Optional[str]) -> str:
    """Minimal preprocessing for RoBERTa-style models.

    Only replaces @mentions with "@user" and URLs with "http", then strips
    surrounding whitespace. Case and emoji are kept: RoBERTa is case-sensitive
    and the Cardiff NLP Twitter models were trained with this convention.
    """
    if not isinstance(text, str):
        return ""
    text = _USER_RE.sub("@user", text)
    text = _URL_RE.sub("http", text)
    return text.strip()
