"""Module 4 (second half) - turn what the user wrote into values the chatbot can use.

The NER model (src/ner/extract.py) only says WHERE the places and dates are. These rules say WHAT they mean:

    extract_slots("nheb billet men tunis l jarba ba3d ghodwa l 2 nfar b 300 dt")
    -> places:     tunis -> Tunis (origin), jarba -> Djerba (destination)
       dates:      ba3d ghodwa -> 2026-11-23
       counts:     people = 2
       currencies: TND, amount 300

Strategy for everything: clean the text -> exact lookup in our lexicons -> fuzzy match -> if unsure, mark it
"ask" so the chatbot asks the user; if nothing is found, the chatbot asks for the missing information.
No slang list (user decision): unknown expressions are not guessed.
"""

import calendar
import difflib
import re
from datetime import date, timedelta

import dateparser

from src.data_gen.entity_matcher import build_date_index, build_place_index, norm_text
from src.data_gen.fill_templates import AIRLINES, CURRENCIES, PLACES
from src.labels import LANGS

WORD_RE = re.compile(r"\w+|[^\w\s]")
ACCEPT, ASK = 0.85, 0.80          # fuzzy thresholds: >= ACCEPT accepted, between ASK and ACCEPT -> ask the user
                                  # (0.80, not 0.75: 'La Marsa' vs 'Marseille' scores 0.77 and must NOT be asked)


def clean(text):
    """lowercase, no accents, squash letters repeated 3+ times (dinarrr -> dinar), single spaces"""
    text = re.sub(r"(\w)\1{2,}", r"\1", norm_text(text))
    return re.sub(r"\s+", " ", text).strip()


def best_match(text, candidates):
    """Closest candidate by difflib similarity (0..1) -> (candidate, score)."""
    best, score = None, 0.0
    for c in candidates:
        s = difflib.SequenceMatcher(None, text, c).ratio()
        if s > score:
            best, score = c, s
    return best, score


# ============================================================================ places

PLACE_INDEX = build_place_index()                    # cleaned spelling -> canonical name
ARTICLES = ("el ", "la ", "le ", "the ")


def normalize_place(text):
    """-> {"canonical", "score", "status"}; status: ok | ask (confirm with the user) | unknown (not in our data)"""
    c = clean(text)
    forms = [c] + [c[len(a):] for a in ARTICLES if c.startswith(a)]
    for f in forms:
        if f in PLACE_INDEX:
            return {"canonical": PLACE_INDEX[f], "score": 1.0, "status": "ok"}
    spelling, score = max((best_match(f, PLACE_INDEX) for f in forms), key=lambda t: t[1])
    if score >= ACCEPT:
        return {"canonical": PLACE_INDEX[spelling], "score": round(score, 2), "status": "ok"}
    if score >= ASK:
        return {"canonical": PLACE_INDEX[spelling], "score": round(score, 2), "status": "ask"}
    return {"canonical": None, "score": round(score, 2), "status": "unknown"}


ORIGIN_WORDS = {"men", "mel", "mn", "from", "de", "du", "d", "depuis"}
DEST_WORDS = {"l", "lel", "to", "a", "pour", "vers", "ila", "towards"}
IN_WORDS = {"f", "fi", "fel", "in", "en", "at", "au", "aux", "dans"}


def place_roles(text, places):
    """origin / destination / location from the small word just before each place."""
    words = [(m.group(), m.start()) for m in WORD_RE.finditer(text)]
    for p in places:
        before = [clean(w) for w, s in words if s < p["start"] and w not in ("'", "’", "-")]
        prev = before[-1] if before else ""
        p["role"] = ("origin" if prev in ORIGIN_WORDS else "destination" if prev in DEST_WORDS
                     else "location" if prev in IN_WORDS else None)
    unknown = [p for p in places if p["role"] is None]
    if len(places) == 2 and len(unknown) == 2:              # keyword style: "vol tunis paris"
        places[0]["role"], places[1]["role"] = "origin", "destination"
    elif len(places) == 2 and len(unknown) == 1:             # "vol tunis l paris" -> tunis is the origin
        other = [p for p in places if p["role"] is not None][0]["role"]
        unknown[0]["role"] = "origin" if other == "destination" else "destination" if other == "origin" else "location"
    for p in places:
        if p["role"] is None:
            p["role"] = "location"
    return places


# ============================================================================ dates

DATE_INDEX = {}                                       # cleaned phrase -> rule, all 3 languages together
for _lang in LANGS:
    for _phrase, (_s, _e, _rule) in build_date_index(_lang).items():
        DATE_INDEX.setdefault(clean(_phrase), _rule)
        DATE_INDEX.setdefault(clean(_phrase[_s:_e]), _rule)

TIME_OF_DAY = {"morning", "afternoon", "evening", "night", "matin", "apres", "midi", "soir", "nuit", "soiree",
               "sbe7", "sbah", "3chiya", "l3chiya", "lil", "ellil", "fel", "fi", "el"}   # "ghodwa fel lil" -> ghodwa
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
                "ten": 10, "un": 1, "une": 1, "deux": 2, "trois": 3, "quatre": 4, "cinq": 5, "sept": 7, "huit": 8,
                "neuf": 9, "dix": 10, "wa7ed": 1, "zouz": 2, "thletha": 3, "tletha": 3, "arba3a": 4, "khamsa": 5,
                "sitta": 6, "sab3a": 7, "thmanya": 8, "tes3a": 9, "3achra": 10}
MONTHS = {}                                           # cleaned month word -> month number, all languages
from src.data_gen.fill_templates import DATES as _DATES   # noqa: E402
for _lang in LANGS:
    for _i, _m in enumerate(_DATES["months"][_lang]):
        MONTHS[clean(_m)] = _i + 1
for _i, _m in enumerate(_DATES["months_tn_dialect_variants"]):
    MONTHS.setdefault(clean(_m), _i + 1)

SOON = {"this week", "cette semaine", "el jem3a hedhi", "had el jem3a", "les prochains jours",
        "the next few days", "next few days", "ces jours", "prochainement", "soon"}
# Fixed-date holidays only. Ramadan / Eid move every year (Islamic calendar): not guessed, the bot asks.
HOLIDAYS = {"christmas": (12, 25), "noel": (12, 25), "new year": (1, 1), "nouvel an": (1, 1), "ras el 3am": (1, 1)}
SEASONS = {"summer": 7, "ete": 7, "sif": 7, "winter": 1, "hiver": 1, "chta": 1,
           "spring": 4, "printemps": 4, "rbi3": 4, "autumn": 10, "fall": 10, "automne": 10, "5rif": 10}


def to_number(word):
    w = clean(word)
    return int(w) if w.isdigit() else NUMBER_WORDS.get(w)


def add_months(d, n):
    y, m = divmod(d.month - 1 + n, 12)
    y, m = d.year + y, m + 1
    return date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def next_date(today, month, day):
    """The next (month, day) from today on (this year if not passed yet, otherwise next year)."""
    d = date(today.year, month, min(day, calendar.monthrange(today.year, month)[1]))
    return d if d >= today else date(today.year + 1, month, min(day, calendar.monthrange(today.year + 1, month)[1]))


def resolve_rule(rule, today):
    """Lexicon rule ('+1d', 'weekday:4', 'md:12-20', ...) -> (date, precision)."""
    if rule == "today":
        return today, "day"
    if m := re.fullmatch(r"\+(\d+)([dwm])", rule):
        n, unit = int(m.group(1)), m.group(2)
        return (today + timedelta(days=n), "day") if unit == "d" else \
               (today + timedelta(weeks=n), "week") if unit == "w" else (add_months(today, n), "month")
    if rule in ("weekend", "next_weekend"):
        sat = today + timedelta(days=(5 - today.weekday()) % 7) if today.weekday() != 6 else today
        return (sat if rule == "weekend" else sat + timedelta(days=7)), "weekend"
    if m := re.fullmatch(r"weekday:(\d)", rule):
        ahead = (int(m.group(1)) - today.weekday()) % 7 or 7
        return today + timedelta(days=ahead), "day"
    if rule == "end_of_month":
        return date(today.year, today.month, calendar.monthrange(today.year, today.month)[1]), "day"
    if m := re.fullmatch(r"season:(\w+)", rule):
        return next_date(today, SEASONS.get(m.group(1), 7), 1), "season"
    if m := re.fullmatch(r"md:(\d\d)-(\d\d)", rule):
        return next_date(today, int(m.group(1)), int(m.group(2))), "day"
    if m := re.fullmatch(r"month:(\d+)", rule):
        return next_date(today, int(m.group(1)), 1), "month"
    return None, None


def normalize_date(text, today=None):
    """-> {"date": "YYYY-MM-DD", "precision", "how"} or None (the chatbot then asks for the date)."""
    today = today or date.today()
    c = clean(text)
    plain = re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", c)).strip()       # no punctuation: "l'ete" -> "l ete"
    short = " ".join(w for w in plain.split() if w not in TIME_OF_DAY) or plain   # "demain matin" -> "demain"
    for form in (c, short):
        if form in DATE_INDEX:                                               # 1) our lexicon
            d, precision = resolve_rule(DATE_INDEX[form], today)
            return {"date": d.isoformat(), "precision": precision, "how": "lexicon"}
    if m := re.fullmatch(r"(?:ba3d|dans|in) (\w+) (iyem|ayem|jours|days|jem3at|semaines|weeks|chhour|mois|months)", short):
        n = to_number(m.group(1))                                            # 2) "ba3d 3 iyem", "dans trois jours"
        if n:
            unit = m.group(2)
            d = today + timedelta(days=n) if unit in ("iyem", "ayem", "jours", "days") else \
                today + timedelta(weeks=n) if unit in ("jem3at", "semaines", "weeks") else add_months(today, n)
            return {"date": d.isoformat(), "precision": "day", "how": "rule"}
    if m := re.fullmatch(r"(fin|a5er|late|end of|debut|awel|early|beginning of) (\w+)", short):
        month = MONTHS.get(m.group(2))                                       # 3) "fin avril", "a5er avril"
        if month:
            day = 3 if m.group(1) in ("debut", "awel", "early", "beginning of") else 25
            return {"date": next_date(today, month, day).isoformat(), "precision": "period", "how": "rule"}
    if plain in SOON or short in SOON:                                       # 4) "this week", "el jem3a hedhi"
        return {"date": today.isoformat(), "precision": "week", "how": "rule"}
    if plain in HOLIDAYS:                                                    # fixed-date holidays only
        month, day = HOLIDAYS[plain]
        return {"date": next_date(today, month, day).isoformat(), "precision": "day", "how": "rule"}
    if m := re.fullmatch(r"(?:nhar|le|the|on the) (\d{1,2})(?:st|nd|rd|th|er)?", plain):
        day = int(m.group(1))                                                # day only: "le 5", "nhar 5", "the 5th"
        if 1 <= day <= 31:
            month = today.month if day >= today.day else (today.month % 12) + 1
            year = today.year if (month >= today.month) else today.year + 1
            day = min(day, calendar.monthrange(year, month)[1])
            return {"date": date(year, month, day).isoformat(), "precision": "day", "how": "rule"}
    words = plain.split()
    season = next((SEASONS[w] for w in words if w in SEASONS), None)
    if season and len(words) <= 3:                                           # 5) "sif ejjey", "el chta hedhi"
        return {"date": next_date(today, season, 1).isoformat(), "precision": "season", "how": "rule"}
    keys = [k for k in DATE_INDEX if len(k) >= 4]
    phrase, score = max(best_match(plain, keys), best_match(short, keys), key=lambda t: t[1])
    if score >= ACCEPT:                                                      # 6) typos: "ghodwaa", "8odwaa"
        d, precision = resolve_rule(DATE_INDEX[phrase], today)
        return {"date": d.isoformat(), "precision": precision, "how": "fuzzy"}
    parsed = dateparser.parse(text, languages=["fr", "en"],                  # 7) general French / English dates
                              settings={"RELATIVE_BASE": _midnight(today), "PREFER_DATES_FROM": "future"})
    if parsed:
        return {"date": parsed.date().isoformat(), "precision": "day", "how": "dateparser"}
    return None


RELATIVE_DATE_RE = re.compile(r"\b(?:ba3d|dans|in) \w+ (?:iyem|ayem|jours|days|jem3at|semaines|weeks|chhour|mois|months)\b")


def _midnight(d):
    from datetime import datetime
    return datetime(d.year, d.month, d.day)


# ============================================================================ currencies and amounts

CURRENCY_INDEX = {}                                   # cleaned spelling -> code
for _code, _by_lang in CURRENCIES.items():
    CURRENCY_INDEX[_code.lower()] = _code
    for _spellings in _by_lang.values():
        for _s in _spellings:
            CURRENCY_INDEX.setdefault(clean(_s), _code)
CURRENCY_INDEX.update({"€": "EUR", "$": "USD", "£": "GBP", "dollar": "USD", "dollars": "USD", "pound": "GBP",
                       "pounds": "GBP", "livre": "GBP", "livres": "GBP"})
AMBIGUOUS = {"USD": {"Canada": "CAD"}, "GBP": {"Turkey": "TRY"}}   # plain "dollar" in Montreal -> CAD, etc.
NUMBER_RE = re.compile(r"^\d+(?:[.,]\d+)?k?$")


def parse_amount(token):
    t = token.lower().replace(",", ".")
    if not NUMBER_RE.match(t):
        return to_number(token)
    return float(t[:-1]) * 1000 if t.endswith("k") else float(t)


def find_currencies(text, countries=()):
    """-> [{"code", "text", "amount"}] in order of appearance (for exchange: 1st = from, 2nd = to)."""
    tokens = [m.group() for m in WORD_RE.finditer(text)]
    found, i = [], 0
    single_words = [k for k in CURRENCY_INDEX if " " not in k and len(k) >= 4]
    while i < len(tokens):
        hit = None
        for n in (3, 2, 1):                                           # longest phrase first: "dinars tunisiens"
            phrase = clean(" ".join(tokens[i:i + n]))
            if i + n <= len(tokens) and phrase in CURRENCY_INDEX:
                hit = (CURRENCY_INDEX[phrase], n, phrase)
                break
        if not hit and len(clean(tokens[i])) >= 4 and not clean(tokens[i]).isdigit():
            spelling, score = best_match(clean(tokens[i]), single_words)   # typos: "dinarrr", "eroo"
            if score >= ACCEPT:
                hit = (CURRENCY_INDEX[spelling], 1, spelling)
        if hit:
            code, n, phrase = hit
            if phrase in ("dollar", "dollars", "pound", "pounds", "livre", "livres"):
                for country in countries:                             # ambiguous: follow the place's country
                    code = AMBIGUOUS.get(code, {}).get(country, code)
            amount = None
            for j in (i - 1, i + n):                                  # a number right before or after
                if 0 <= j < len(tokens):
                    amount = parse_amount(tokens[j])
                    if amount is not None:
                        break
            found.append({"code": code, "text": " ".join(tokens[i:i + n]), "amount": amount})
            i += n
        else:
            i += 1
    return found


# ============================================================================ counts and airlines

UNITS = {
    "nights": {"nuit", "nuits", "night", "nights", "lila", "lyeli", "liali", "leyla"},
    "days": {"jour", "jours", "day", "days", "ayem", "iyem", "youm", "nhar"},
    "people": {"personne", "personnes", "people", "persons", "person", "adultes", "adults", "nfar", "nas", "ashkhas"},
    "kg": {"kg", "kilo", "kilos", "kilogrammes", "kilograms"},
}


def find_counts(text):
    """-> {"nights": 3, "people": 2, ...} from 'number + unit' (or 'a7na N' = we are N)."""
    tokens = [clean(m.group()) for m in WORD_RE.finditer(text)]
    counts = {}
    for i, tok in enumerate(tokens):
        n = to_number(tok)
        if n is None or i + 1 >= len(tokens):
            continue
        for unit, words in UNITS.items():
            if tokens[i + 1] in words:
                counts.setdefault(unit, n)
        if i > 0 and tokens[i - 1] in ("a7na", "on est", "we are"):
            counts.setdefault("people", n)
    return counts


AIRLINE_INDEX = {clean(a): a for a in AIRLINES}


def find_airlines(text):
    tokens = [m.group() for m in WORD_RE.finditer(text) if m.group().isalnum()]
    found = []
    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            phrase = clean(" ".join(tokens[i:i + n]))
            if len(phrase) < 5:
                continue
            name, score = best_match(phrase, AIRLINE_INDEX)
            if score >= ACCEPT and AIRLINE_INDEX[name] not in found:
                found.append(AIRLINE_INDEX[name])
    return found


# ============================================================================ everything together

def extract_slots(text, today=None, entities=None):
    """All the information the chatbot needs from one message. `entities` = NER output (computed if not given)."""
    if entities is None:
        from src.ner.extract import extract_entities
        entities = extract_entities(text)
    places = []
    for e in entities:
        if e["label"] == "LOC":
            places.append({"text": e["text"], "start": e["start"], **normalize_place(e["text"])})
    place_roles(text, places)
    dates = []
    for e in entities:
        if e["label"] == "DATE":
            dates.append({"text": e["text"], **(normalize_date(e["text"], today) or {"date": None})})
    if not dates:                                       # safety net: the NER is weaker on unusual date phrases
        for m in RELATIVE_DATE_RE.finditer(clean(text)):
            found = normalize_date(m.group(), today)
            if found:
                dates.append({"text": m.group(), **found, "how": "rule (no NER date)"})
                break
    countries = [PLACES[p["canonical"]]["country"] for p in places if p["canonical"]]
    slots = {"places": places, "dates": dates, "currencies": find_currencies(text, countries),
             "counts": find_counts(text), "airlines": find_airlines(text)}
    slots["to_confirm"] = [p["text"] for p in places if p["status"] == "ask"] + \
                          [d["text"] for d in dates if d["date"] is None]
    return slots


if __name__ == "__main__":                 # quick manual check:  python -m src.ner.rules
    import json
    TODAY = date(2026, 11, 21)
    for msg in ["nheb billet men tunis l jarba ba3d ghodwa l 2 nfar b 300 dt",
                "vol djrba paris dmain matin",
                "Je veux changer 500 eurooo en dinars tunisiens",
                "how much is a hotel in Montreal for 3 nights, around 200 dollars?",
                "fama bac men sfax l kerkennah ba3d 3 iyem?",
                "s5ouna f tamerza fi a5er avril?",
                "baggage allowance with tunisaire, my bag is 25 kg"]:
        print(msg)
        print("   ", json.dumps(extract_slots(msg, today=TODAY), ensure_ascii=False))
