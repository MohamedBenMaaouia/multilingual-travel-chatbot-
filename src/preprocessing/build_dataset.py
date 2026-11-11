"""Step 1.5 - Merge, clean, check, split and report the dataset.

Inputs : data/generated/template_queries.jsonl  (template rows)
         data/generated/external_queries.jsonl  (GPT rows)
Outputs: data/processed/dataset.jsonl          every row, with its "split" field
         data/processed/{train,val,test}.jsonl  the three sets every model will use
         data/processed/balance_report.md       counts for the datasheet + leakage checks

What happens:
  1. merge both sources
  2. safe cleaning that learns nothing from the data (so it is fine to do before the split):
     unify apostrophes, collapse repeated spaces, trim. Entity offsets are recomputed so every span
     still points at the right characters. NO lowercasing here: each model normalizes its own input.
  3. drop duplicates across sources (same text ignoring case/accents/punctuation)
  4. template-aware stratified split 70/15/15:
       - a "group" = all rows from the same template (GPT rows are each their own group)
       - groups never leave their (language, subcategory) cell, so we split cell by cell
       - groups are shuffled (seed 42) and each goes to the set that is furthest below its share
  5. write the files and a balance report, and verify that no template / sentence is in two sets

Run from the repo root:
    python -m src.preprocessing.build_dataset
"""

import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.labels import LANGS, SUBCATEGORIES
from src.data_gen.fill_templates import dedup_key, mask

ROOT = Path(__file__).resolve().parents[2]
SOURCES = [ROOT / "data" / "generated" / "template_queries.jsonl",
           ROOT / "data" / "generated" / "external_queries.jsonl"]
OUT_DIR = ROOT / "data" / "processed"

SEED = 42
SHARES = {"train": 0.70, "val": 0.15, "test": 0.15}
NON_TRAVEL = "non_travel"   # name used for the non-travel "cell" in stratification and reports


# --------------------------------------------------------------------------- cleaning

def clean_text_and_spans(text, entities):
    """Collapse whitespace, trim, unify apostrophes, and move entity offsets accordingly.
    We build the new text character by character and remember where each old character went."""
    text = text.replace("’", "'").replace("‘", "'")
    new_chars, old_to_new = [], {}
    for i, c in enumerate(text):
        if c.isspace():
            if not new_chars or new_chars[-1] == " ":
                old_to_new[i] = len(new_chars)          # extra space removed: points at next char
                continue
            c = " "
        old_to_new[i] = len(new_chars)
        new_chars.append(c)
    new_text = "".join(new_chars)
    lead = len(new_text) - len(new_text.lstrip())
    new_text = new_text.strip()

    new_ents = []
    for e in entities:
        s = old_to_new[e["start"]] - lead
        end = old_to_new[e["end"] - 1] + 1 - lead
        new_ents.append({**e, "start": s, "end": end, "value": new_text[s:end]})
    return new_text, new_ents


def check_spans(row):
    """Every entity must point at non-empty, trimmed text, inside the sentence, without overlaps."""
    prev_end = 0
    for e in sorted(row["entities"], key=lambda e: e["start"]):
        v = row["text"][e["start"]:e["end"]]
        if not v or v != v.strip() or e["start"] < prev_end or e["end"] > len(row["text"]):
            return False
        prev_end = e["end"]
    return True


# --------------------------------------------------------------------------- split

def cell_of(row):
    return (row["lang"], row["subcategory"] or NON_TRAVEL)


def template_aware_split(rows, rng):
    """Assign a 'split' to every row. Groups (same template) always stay together."""
    by_cell = defaultdict(lambda: defaultdict(list))
    for r in rows:
        group = r["template_id"] or r["id"]          # GPT rows: each row is its own group
        by_cell[cell_of(r)][group].append(r)

    for cell in sorted(by_cell):
        groups = list(by_cell[cell].values())
        rng.shuffle(groups)
        n_total = sum(len(g) for g in groups)
        filled = {s: 0 for s in SHARES}
        for g in groups:
            # put this group in the set that is furthest below its target size
            target = max(SHARES, key=lambda s: SHARES[s] * n_total - filled[s])
            for r in g:
                r["split"] = target
            filled[target] += len(g)


# --------------------------------------------------------------------------- report

def table(header, rows_):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(x) for x in r) + " |" for r in rows_]
    return "\n".join(out)


def build_report(rows, n_dropped_dup, n_bad_spans):
    cols = SUBCATEGORIES + [NON_TRAVEL]
    lines = ["# Dataset balance report", "",
             f"Total rows: **{len(rows)}**  (duplicates dropped: {n_dropped_dup}, bad spans dropped: {n_bad_spans})", ""]

    # per language and source
    lines += ["## Rows per language and source", ""]
    src = Counter((r["lang"], r["source"], r["is_travel"]) for r in rows)
    lines.append(table(["lang", "template (travel)", "gpt travel", "gpt non-travel", "total"],
                       [[l, src[(l, "template", True)], src[(l, "llm", True)], src[(l, "llm", False)],
                         sum(1 for r in rows if r["lang"] == l)] for l in LANGS]))

    # language x subcategory grid
    lines += ["", "## Language x subcategory", ""]
    grid = Counter(cell_of(r) for r in rows)
    body = [[c] + [grid[(l, c)] for l in LANGS] + [sum(grid[(l, c)] for l in LANGS)] for c in cols]
    lines.append(table(["subcategory"] + LANGS + ["total"], body))

    # per split
    lines += ["", "## Split sizes", ""]
    sp = Counter((r["split"], r["lang"]) for r in rows)
    body = [[s] + [sp[(s, l)] for l in LANGS] + [sum(sp[(s, l)] for l in LANGS),
            f"{100 * sum(sp[(s, l)] for l in LANGS) / len(rows):.1f}%"] for s in SHARES]
    lines.append(table(["split"] + LANGS + ["total", "share"], body))

    lines += ["", "## Test rows per language x subcategory (smallest cells matter most)", ""]
    tg = Counter(cell_of(r) for r in rows if r["split"] == "test")
    lines.append(table(["subcategory"] + LANGS, [[c] + [tg[(l, c)] for l in LANGS] for c in cols]))

    # entities
    lines += ["", "## Entities", ""]
    ent = Counter((r["lang"], e["label"]) for r in rows for e in r["entities"])
    with_any = Counter(r["lang"] for r in rows if r["entities"])
    lines.append(table(["lang", "LOC", "DATE", "rows with ≥1 entity"],
                       [[l, ent[(l, "LOC")], ent[(l, "DATE")], with_any[l]] for l in LANGS]))

    # leakage checks
    tpl_splits = defaultdict(set)
    key_splits = defaultdict(set)
    for r in rows:
        if r["template_id"]:
            tpl_splits[r["template_id"]].add(r["split"])
        key_splits[dedup_key(r["text"])].add(r["split"])
    shared_tpl = sum(1 for s in tpl_splits.values() if len(s) > 1)
    shared_txt = sum(1 for s in key_splits.values() if len(s) > 1)
    lines += ["", "## Leakage checks", "",
              f"- Templates appearing in more than one split: **{shared_tpl}** (must be 0)",
              f"- Identical sentences appearing in more than one split: **{shared_txt}** (must be 0)", ""]
    return "\n".join(lines), shared_tpl, shared_txt


# --------------------------------------------------------------------------- main

def main():
    rows = []
    for path in SOURCES:
        with open(path, encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]

    # 1-2. clean + check spans
    cleaned, n_bad = [], 0
    for r in rows:
        text, ents = clean_text_and_spans(r["text"], r["entities"])
        r = {**r, "text": text, "entities": ents, "text_masked": mask(text, ents)}
        if check_spans(r):
            cleaned.append(r)
        else:
            n_bad += 1

    # 3. drop duplicates across sources (template rows come first, so they are the ones kept)
    seen, unique = set(), []
    for r in cleaned:
        k = dedup_key(r["text"])
        if k not in seen:
            seen.add(k)
            unique.append(r)
    n_dup = len(cleaned) - len(unique)

    # final ids, and a fixed field order so the files are easy to read
    fields = ["id", "text", "lang", "is_travel", "category", "subcategory", "entities", "text_masked",
              "source", "template_id", "llm_model", "verified", "multi_intent"]
    final = []
    for i, r in enumerate(unique, 1):
        r["id"] = f"q_{i:06d}"
        r.setdefault("llm_model", None)
        final.append({k: r.get(k) for k in fields})

    # 4. split
    template_aware_split(final, random.Random(SEED))

    # 5. write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "dataset.jsonl", "w", encoding="utf-8") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    for s in SHARES:
        with open(OUT_DIR / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for r in final:
                if r["split"] == s:
                    f.write(json.dumps({k: v for k, v in r.items() if k != "split"}, ensure_ascii=False) + "\n")

    report, shared_tpl, shared_txt = build_report(final, n_dup, n_bad)
    (OUT_DIR / "balance_report.md").write_text(report, encoding="utf-8")

    sizes = Counter(r["split"] for r in final)
    print(f"Rows: {len(final)} (dropped {n_dup} duplicates, {n_bad} bad spans)")
    print("Split sizes:", {s: f"{sizes[s]} ({100 * sizes[s] / len(final):.1f}%)" for s in SHARES})
    print(f"Leakage checks: templates in >1 split = {shared_tpl}, same sentence in >1 split = {shared_txt}")
    print("Report: data/processed/balance_report.md")


if __name__ == "__main__":
    main()
