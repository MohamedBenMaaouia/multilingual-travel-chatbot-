"""Module 1 - Language detection, as used by the chatbot.

    from src.lang_detect.detect import detect_language
    detect_language("Chnowa el ta9s f Jarba ghodwa?")   ->  ("tn_latin", 0.99...)

Returns the language (en / fr / tn_latin) and the model's confidence (probability between 0 and 1).
Deciding what to do when the confidence is low (e.g. keep the conversation's language for a message
like "paris ?") is the orchestrator's job, not this function's.

The model is trained in notebooks/01_lang_detect_fasttext.ipynb
(whole words, lr=1.0, 10 epochs; val accuracy 99.71%, challenge set 59/60).
"""

from functools import lru_cache
from pathlib import Path

import fasttext

from src.preprocessing.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "lang_detect" / "fasttext_lang.bin"


@lru_cache(maxsize=1)          # load the model the first time only, then reuse it
def _load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{MODEL_PATH} not found. Train and save it by running notebooks/01_lang_detect_fasttext.ipynb "
            "(models/ is not stored in git).")
    return fasttext.load_model(str(MODEL_PATH))


def detect_language(text: str) -> tuple[str | None, float]:
    """Return (language, confidence). An empty message returns (None, 0.0)."""
    cleaned = normalize(text)              # SAME cleaning as the training data
    if not cleaned:
        return None, 0.0
    labels, probs = _load_model().predict(cleaned, k=1)
    confidence = min(float(probs[0]), 1.0)   # fastText can return 1.00001 (rounding quirk)
    return labels[0].replace("__label__", ""), confidence


if __name__ == "__main__":                 # quick manual check:  python -m src.lang_detect.detect
    for msg in ["Chnowa el ta9s f Jarba ghodwa?", "Quel temps fait-il à Djerba demain ?",
                "What's the weather in Djerba tomorrow?", "nheb billet l paris", "paris ?", "   "]:
        print(f"{msg!r:45} -> {detect_language(msg)}")
