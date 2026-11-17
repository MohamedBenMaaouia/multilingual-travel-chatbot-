"""Module 1, step 3 - Turn our dataset into fastText's training format.

fastText reads one example per line: the label with the prefix __label__, then the text.
    __label__tn_latin chnowa el ta9s f jarba ghodwa ?

Decisions:
  * we use the original `text` (cleaned with normalize()), not `text_masked`: at run time the detector
    sees real messages, so it learns from real messages
  * training lines are SHUFFLED with a fixed seed: fastText learns in file order, and our file is grouped
    by language, so without shuffling the model would be biased toward whatever language came last
  * only train and val are written. The test set stays untouched until the final evaluation.

Run from the repo root:
    python -m src.lang_detect.prepare_data
Output: data/processed/fasttext_lang/{train,val}.txt
"""

import json
import random
from collections import Counter
from pathlib import Path

from src.preprocessing.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
IN_DIR = ROOT / "data" / "processed"
OUT_DIR = ROOT / "data" / "processed" / "fasttext_lang"
SEED = 42


def to_fasttext_line(row):
    return f"__label__{row['lang']} {normalize(row['text'])}"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val"]:
        with open(IN_DIR / f"{split}.jsonl", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]

        lines = [to_fasttext_line(r) for r in rows]
        if split == "train":
            random.Random(SEED).shuffle(lines)   # mix the languages (see the docstring)

        with open(OUT_DIR / f"{split}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        counts = Counter(r["lang"] for r in rows)
        print(f"{split}: {len(lines)} lines -> {OUT_DIR.relative_to(ROOT) / (split + '.txt')}   {dict(counts)}")


if __name__ == "__main__":
    main()
