"""Shared text cleaning used by the models.

The SAME function must be used on the training data and on every live message in the chatbot.
If training and real use cleaned text differently, the model would see pieces it never learned
("train/serve skew").

What it does:   NFC unicode form, lowercase, straight apostrophes, spaces around punctuation, tidy spaces.
What it keeps:  digits (3/7/9/5/8 are letters in Tunisian Arabizi), accents (é, è are French clues),
                apostrophes and hyphens inside words (l'hôtel, après-demain).
"""

import re
import unicodedata

PUNCTUATION = r"([?!.,;:()\"«»…])"


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text)        # 1. one single form for accented letters
    text = text.lower()                              # 2. Jarba -> jarba
    text = text.replace("’", "'").replace("‘", "'")  # 3. curly -> straight apostrophe
    text = re.sub(PUNCTUATION, r" \1 ", text)        # 4. "jarba?" -> "jarba ?"
    text = re.sub(r"\s+", " ", text).strip()         # 5. collapse spaces, trim
    return text
