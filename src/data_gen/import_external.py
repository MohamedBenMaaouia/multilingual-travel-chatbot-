"""Step 1.4 - Import the externally generated queries (GPT) into our dataset format.

Input : data/external/gpt_queries.jsonl   lines like {"lang": "...", "subcategory": "... or non_travel", "text": "..."}
Output: data/generated/external_queries.jsonl   same schema as template_queries.jsonl
        data/generated/external_rejected.jsonl  every dropped line with the reason (for the datasheet)

What it does, line by line:
  1. reads the file robustly (repairs an object split over several lines, skips ``` fences / blank lines)
  2. cleans the text (extra spaces, surrounding quotes)
  3. drops: Arabic letters in tn_latin, too short/long, duplicates (also against template rows),
     and travel rows where the matcher finds a date/place it cannot label
  4. labels LOC / DATE entities with the lexicon-based Matcher (spans never come from GPT)
  5. caps each travel cell so templates + GPT <= TARGET (87); non-travel rows are all kept

Run from the repo root:
    python -m src.data_gen.import_external
"""

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from src.labels import LANGS, SUBCAT_TO_CAT
from src.data_gen.fill_templates import dedup_key, mask
from src.data_gen.entity_matcher import ARABIC_SCRIPT, Matcher

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "external" / "gpt_queries.jsonl"
TEMPLATE_ROWS = ROOT / "data" / "generated" / "template_queries.jsonl"
OUT = ROOT / "data" / "generated" / "external_queries.jsonl"
REJECTED = ROOT / "data" / "generated" / "external_rejected.jsonl"

TARGET = 87          # rows per travel cell (templates + external)
SEED = 42
sys.stdout.reconfigure(encoding="utf-8")


def read_jsonl_robust(path):
    """Yield dicts. If an object is split over several lines, keep adding lines until it parses."""
    buffer = ""
    for line in open(path, encoding="utf-8"):
        s = line.strip()
        if not s or s.startswith("```"):
            continue
        buffer = f"{buffer} {s}" if buffer else s
        try:
            yield json.loads(buffer.rstrip(","))
            buffer = ""
        except json.JSONDecodeError:
            if len(buffer) > 2000:      # give up on garbage so it can't swallow the rest of the file
                print("  skipped unreadable text:", buffer[:80])
                buffer = ""


def clean(text):
    return re.sub(r"\s+", " ", text).strip().strip('"').strip()


def main():
    template_rows = [json.loads(l) for l in open(TEMPLATE_ROWS, encoding="utf-8")]
    have = Counter((r["lang"], r["subcategory"]) for r in template_rows)
    seen = {dedup_key(r["text"]) for r in template_rows}
    matchers = {lang: Matcher(lang) for lang in LANGS}

    accepted = defaultdict(list)     # cell -> rows
    rejected = []
    n_read = 0

    for item in read_jsonl_robust(IN):
        n_read += 1
        lang, sub, text = item.get("lang"), item.get("subcategory"), item.get("text")

        def reject(reason):
            rejected.append({"lang": lang, "subcategory": sub, "text": text, "reason": reason})

        if lang not in LANGS or (sub not in SUBCAT_TO_CAT and sub != "non_travel") or not isinstance(text, str):
            reject("bad fields"); continue
        text = clean(text)
        if lang == "tn_latin" and ARABIC_SCRIPT.search(text):
            reject("arabic script"); continue
        if not 2 <= len(text.split()) <= 35:
            reject("length"); continue
        key = dedup_key(text)
        if key in seen:
            reject("duplicate"); continue

        is_travel = sub != "non_travel"
        ents, reason = matchers[lang].find_entities(text, strict=is_travel)
        if reason:
            reject(reason); continue

        seen.add(key)
        accepted[(lang, sub)].append({
            "text": text,
            "lang": lang,
            "is_travel": is_travel,
            "category": SUBCAT_TO_CAT[sub] if is_travel else None,
            "subcategory": sub if is_travel else None,
            "entities": ents,
            "text_masked": mask(text, ents),
            "source": "llm",
            "template_id": None,
            "llm_model": "gpt",
            "verified": False,
            "multi_intent": False,
        })

    # Cap travel cells so templates + external <= TARGET (random but reproducible choice of which to keep)
    rng = random.Random(SEED)
    rows = []
    for cell, cell_rows in sorted(accepted.items()):
        lang, sub = cell
        if sub != "non_travel":
            room = max(TARGET - have[cell], 0)
            if len(cell_rows) > room:
                rng.shuffle(cell_rows)
                for r in cell_rows[room:]:
                    rejected.append({"lang": lang, "subcategory": sub, "text": r["text"], "reason": "cell full"})
                cell_rows = cell_rows[:room]
        rows.extend(cell_rows)

    for i, r in enumerate(rows, 1):
        r["id"] = f"gpt_{i:06d}"
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({"id": r.pop("id"), **r}, ensure_ascii=False) + "\n")
    with open(REJECTED, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- summary
    print(f"Read {n_read} lines -> kept {len(rows)}, rejected {len(rejected)}")
    print("Rejection reasons:", dict(Counter(r["reason"].split(":")[0] for r in rejected)))
    for lang in LANGS:
        travel = [r for r in rows if r["lang"] == lang and r["is_travel"]]
        non = [r for r in rows if r["lang"] == lang and not r["is_travel"]]
        print(f"  {lang:9s} travel {len(travel):4d} | non-travel {len(non):4d}")
    short = {f"{l}/{s}": have[(l, s)] + len(accepted[(l, s)]) for l in LANGS for s in SUBCAT_TO_CAT
             if have[(l, s)] + len(accepted[(l, s)]) < TARGET}
    print("Travel cells below target (templates + external):", short or "none")


if __name__ == "__main__":
    main()
