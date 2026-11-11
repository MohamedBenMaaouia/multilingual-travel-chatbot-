"""Step 1.3 - Fill query templates with values from the lexicons.

For every (language, subcategory) cell we pick templates, fill each {SLOT} with a random value,
and record exactly where each LOC / DATE entity lands in the final sentence (character offsets).
Because we insert the entities ourselves, the NER labels are exact.

Run from the repo root:
    python -m src.data_gen.fill_templates            (default: 50 queries per cell)
    python -m src.data_gen.fill_templates --per-cell 60

The other ~37 queries per cell (target ~87) come from LLM generation.

Output: data/generated/template_queries.jsonl
"""

import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

from src.labels import LANGS, SUBCAT_TO_CAT

ROOT = Path(__file__).resolve().parents[2]
LEX = ROOT / "data" / "lexicons"
TEMPLATES = ROOT / "data" / "templates"
OUT = ROOT / "data" / "generated" / "template_queries.jsonl"

SEED = 42

# Which slots produce entities, and which places they may use
LOC_SLOTS = {
    "LOC": "any",
    "ORIGIN": "any",
    "LOC_TN": "tunisia",
    "ORIGIN_TN": "tunisia",
    "LOC_ABROAD": "abroad",
}
AMOUNTS = [20, 50, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000]


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


PLACES = load_json(LEX / "places.json")["places"]
DATES = load_json(LEX / "dates.json")
CURRENCIES = load_json(LEX / "currencies.json")["currencies"]
AIRLINES = load_json(LEX / "airlines.json")["airlines"]


# --------------------------------------------------------------------------- helpers

def split_brackets(s):
    """'on [December 20]' -> ('on December 20', 3, 14): the text plus where the entity sits.
    No brackets -> the whole text is the entity."""
    if "[" not in s:
        return s, 0, len(s)
    start = s.index("[")
    end = s.index("]") - 1  # -1 because the '[' before it will be removed
    clean = s.replace("[", "").replace("]", "")
    return clean, start, end


def ordinal(n):
    """1 -> 1st, 2 -> 2nd, 11 -> 11th, 22 -> 22nd"""
    if 11 <= n % 100 <= 13:
        return f"{n}th"
    return f"{n}{ {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th') }"


# --------------------------------------------------------------------------- slot values
# Each make_* function returns (text_to_insert, entity_or_None).
# entity = (label, start_inside_value, end_inside_value, norm)
#   norm = canonical form, useful later to check Module 4's normalization:
#   LOC -> canonical city name ("Djerba"), DATE -> the rule ("+1d", "weekday:4", "md:12-20", "month:12")

def make_place(slot, lang, rng, used):
    region = LOC_SLOTS[slot]
    candidates = [name for name, p in PLACES.items()
                  if (region == "any" or p["region"] == region) and name not in used]
    # Most of the agency's customers leave from Tunisia, so origins are usually Tunisian cities
    if slot == "ORIGIN" and rng.random() < 0.7:
        candidates = [c for c in candidates if PLACES[c]["region"] == "tunisia"]
    name = rng.choice(candidates)
    used.add(name)
    spelling = rng.choice(PLACES[name]["spellings"][lang])
    return spelling, ("LOC", 0, len(spelling), name)


def make_date(lang, rng):
    kind = rng.choices(["relative", "weekday", "absolute"], weights=[5, 2, 3])[0]

    if kind == "relative":
        entry = rng.choice(DATES["relative"][lang])
        raw, norm = entry["text"], entry["rule"]

    elif kind == "weekday":
        k = rng.randrange(7)
        fmt = rng.choice(DATES["weekday_formats"][lang])
        raw, norm = fmt.format(weekday=DATES["weekdays"][lang][k]), f"weekday:{k}"

    else:  # absolute date like "le 20 décembre" or "fi chhar 7"
        fmt = rng.choice(DATES["absolute_formats"][lang])
        day, m = rng.randint(1, 28), rng.randrange(12)
        months = DATES["months"][lang]
        if lang == "tn_latin" and rng.random() < 0.3:
            months = DATES["months_tn_dialect_variants"]
        day_txt = "1er" if (lang == "fr" and day == 1) else str(day)
        raw = fmt.format(day=day_txt, month=months[m], month_num=m + 1, day_ord=ordinal(day))
        norm = f"md:{m + 1:02d}-{day:02d}" if "{day" in fmt else f"month:{m + 1}"

    text, s, e = split_brackets(raw)
    return text, ("DATE", s, e, norm)


def make_currency_pair(rng):
    """Pick two different currencies; usually one of them is the Tunisian dinar."""
    codes = list(CURRENCIES)
    if rng.random() < 0.7:
        other = rng.choice([c for c in codes if c != "TND"])
        pair = ["TND", other]
        rng.shuffle(pair)
        return pair
    return rng.sample([c for c in codes if c != "TND"], 2)


def make_plain(slot, lang, rng, state):
    """Slots that are NOT entities: just words/numbers to make the sentence complete."""
    if slot == "AIRLINE":
        return rng.choice(AIRLINES)
    if slot == "STARS":
        return str(rng.randint(3, 5))
    if slot == "NIGHTS":
        return str(rng.randint(2, 10))
    if slot == "DAYS":
        return str(rng.randint(2, 14))
    if slot == "PEOPLE":
        return str(rng.randint(2, 6))
    if slot == "KG":
        return str(rng.randint(15, 32))
    if slot == "AMOUNT":
        return str(rng.choice(AMOUNTS))
    if slot == "CUR":
        code = rng.choices(list(CURRENCIES), weights=[5 if c == "TND" else 3 if c == "EUR" else 1 for c in CURRENCIES])[0]
        return rng.choice(CURRENCIES[code][lang])
    if slot in ("CUR_FROM", "CUR_TO"):
        if "cur_pair" not in state:
            state["cur_pair"] = make_currency_pair(rng)
        code = state["cur_pair"][0 if slot == "CUR_FROM" else 1]
        return rng.choice(CURRENCIES[code][lang])
    raise ValueError(f"Unknown slot {slot}")


# --------------------------------------------------------------------------- building a query

FR_VOWELS = "aeiouyàâäéèêëîïôöûüœAEIOUYÀÂÄÉÈÊËÎÏÔÖÛÜŒ"


def french_elision(text_so_far, value):
    """'... de ' + 'Istanbul' -> '... d'' so we get "d'Istanbul" (also "d'euros")."""
    if value and value[0] in FR_VOWELS and re.search(r"(^|\s)de $", text_so_far):
        return text_so_far[:-3] + "d'"
    return text_so_far


def fill(template, lang, rng):
    """Turn one template into (text, entities)."""
    text, entities = "", []
    used_places, state = set(), {}

    for part in re.split(r"(\{\w+\})", template):
        m = re.fullmatch(r"\{(\w+)\}", part)
        if not m:
            text += part
            continue
        slot = m.group(1)

        if slot in LOC_SLOTS:
            value, ent = make_place(slot, lang, rng, used_places)
        elif slot == "DATE":
            value, ent = make_date(lang, rng)
        else:
            value, ent = make_plain(slot, lang, rng, state), None

        if lang == "fr":
            text = french_elision(text, value)
        start = len(text)
        text += value
        if ent:
            label, s, e, norm = ent
            entities.append({"start": start + s, "end": start + e, "label": label, "norm": norm})

    return text, entities


def strip_accents(text):
    """é -> e, ç -> c, one character for one character (so entity offsets don't move)."""
    return "".join(unicodedata.normalize("NFD", c)[0] for c in text)


def casual_variation(text, lang, rng):
    """Make some queries look like real chat typing. Only changes that keep the
    character offsets of the entities valid (same length, or edits at the very end)."""
    if lang in ("en", "fr") and rng.random() < 0.3:
        text = text.lower()
    if lang == "fr" and rng.random() < 0.3:
        text = strip_accents(text)
    if text.endswith("?") and rng.random() < (0.4 if lang == "tn_latin" else 0.3):
        text = text[:-1].rstrip()          # drop the question mark
    elif lang == "fr" and text.endswith(" ?") and rng.random() < 0.5:
        text = text[:-2] + "?"             # "Djerba ?" -> "Djerba?"
    return text


def dedup_key(text):
    """Two queries that differ only by case, accents or punctuation count as the same query."""
    return re.sub(r"[^\w]+", " ", strip_accents(text.lower())).strip()


def mask(text, entities):
    """Replace each entity span by <LABEL>:  'meteo jerba ghodwa' -> 'meteo <LOC> <DATE>'"""
    for ent in sorted(entities, key=lambda e: e["start"], reverse=True):
        text = text[:ent["start"]] + f"<{ent['label']}>" + text[ent["end"]:]
    return text


# --------------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-cell", type=int, default=50, help="queries per (language, subcategory)")
    args = parser.parse_args()

    rng = random.Random(SEED)
    rows, n = [], 0

    for lang in LANGS:
        templates = load_json(TEMPLATES / f"{lang}.json")
        for sub, tpls in templates.items():
            if sub.startswith("_"):
                continue
            order = list(range(len(tpls)))
            rng.shuffle(order)
            seen = set()
            for i in range(args.per_cell):
                t_idx = order[i % len(order)]      # cycle through templates so each one is used evenly
                for _attempt in range(30):         # retry if we produce a duplicate sentence
                    text, ents = fill(tpls[t_idx], lang, rng)
                    text = casual_variation(text, lang, rng)
                    if dedup_key(text) not in seen:
                        break
                else:
                    continue  # this template can't produce new sentences any more; skip
                seen.add(dedup_key(text))

                for e in ents:
                    e["value"] = text[e["start"]:e["end"]]   # the exact surface text of the entity
                n += 1
                rows.append({
                    "id": f"tpl_{n:06d}",
                    "text": text,
                    "lang": lang,
                    "is_travel": True,
                    "category": SUBCAT_TO_CAT[sub],
                    "subcategory": sub,
                    "entities": [{k: e[k] for k in ("start", "end", "label", "value", "norm")} for e in ents],
                    "text_masked": mask(text, ents),
                    "source": "template",
                    "template_id": f"{lang}/{sub}/{t_idx}",
                    "verified": False,
                    "multi_intent": False,
                })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---- small summary
    print(f"Wrote {len(rows)} queries to {OUT.relative_to(ROOT)}")
    for lang in LANGS:
        lr = [r for r in rows if r["lang"] == lang]
        n_loc = sum(any(e["label"] == "LOC" for e in r["entities"]) for r in lr)
        n_date = sum(any(e["label"] == "DATE" for e in r["entities"]) for r in lr)
        print(f"  {lang:9s} {len(lr):4d} rows | with LOC: {n_loc} | with DATE: {n_date}")


if __name__ == "__main__":
    main()
