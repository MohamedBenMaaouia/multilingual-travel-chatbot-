"""Find LOC / DATE entities in free text (text we did not generate from templates, e.g. GPT-written queries).

Every span comes from our lexicons: we match known place spellings and date phrases.
If a text still contains something that looks like a date or a place we could NOT match
(a month name, 'tomorrow' in an odd form, a country...), find_entities() returns a reason to reject it,
so no accepted row has a missing label.

    m = Matcher("tn_latin")
    ents, reason = m.find_entities("chnowa el ta9s f jarba ghodwa?")
"""

import re

from src.data_gen.fill_templates import DATES, PLACES, ordinal, split_brackets, strip_accents


def norm_text(s):
    """Lowercase + remove accents + unify apostrophes. Keeps the same length, so offsets stay valid."""
    return strip_accents(s.lower()).replace("’", "'")


def build_date_index(lang):
    """Every date phrase we know for this language -> (entity start, entity end, norm).
    Keys are full surfaces (e.g. 'in december'); the entity may be only part of it ('december')."""
    idx = {}

    def add(raw, norm):
        clean, s, e = split_brackets(raw)
        idx.setdefault(norm_text(clean), (s, e, norm))

    for entry in DATES["relative"][lang]:
        add(entry["text"], entry["rule"])
        clean, s, e = split_brackets(entry["text"])
        add(clean[s:e], entry["rule"])                       # also without the preposition

    for k, wd in enumerate(DATES["weekdays"][lang]):
        for fmt in DATES["weekday_formats"][lang]:
            add(fmt.format(weekday=wd), f"weekday:{k}")
        if lang in ("en", "fr"):
            add(wd, f"weekday:{k}")                          # bare 'Monday' / 'lundi'

    month_lists = [DATES["months"][lang]]
    if lang == "tn_latin":
        month_lists.append(DATES["months_tn_dialect_variants"])
    formats = list(DATES["absolute_formats"][lang])
    if lang == "en":
        formats += ["[{month} {day_ord}]", "[the {day_ord} of {month}]", "[{day_ord} {month}]"]
    for months in month_lists:
        for m, month in enumerate(months):
            for fmt in formats:
                if "{day" not in fmt:                        # month only: keep the preposition ('in may')
                    add(fmt.format(month=month, month_num=m + 1), f"month:{m + 1}")
                    continue
                for day in range(1, 32):
                    day_txts = [str(day)] + (["1er"] if (lang == "fr" and day == 1) else [])
                    for dt in day_txts:
                        raw = fmt.format(day=dt, month=month, month_num=m + 1, day_ord=ordinal(day))
                        add(raw, f"md:{m + 1:02d}-{day:02d}")
                        clean, s, e = split_brackets(raw)
                        add(clean[s:e], f"md:{m + 1:02d}-{day:02d}")  # entity part alone ('december 20')
    return idx


def build_place_index():
    """Every spelling in every language -> canonical name (a French query may write 'Jerba', etc.)."""
    idx = {}
    for name, p in PLACES.items():
        for spellings in p["spellings"].values():
            for sp in spellings:
                idx[norm_text(sp)] = name
    return idx


def compile_alternation(keys):
    keys = sorted(keys, key=len, reverse=True)               # longest first: 'ba3d ghodwa' before 'ghodwa'
    return re.compile(r"(?<![\w'])(" + "|".join(re.escape(k) for k in keys) + r")(?![\w])")


# Words/patterns that signal a date or a place we did NOT label -> the row is rejected.
SUSPICIOUS_DATE = re.compile(r"(?<!\w)(" + "|".join([
    r"tomorrow", r"today", r"tonight", r"yesterday", r"weekend", r"week-end", r"demain", r"aujourd'hui", r"hier",
    r"ghodwa", r"ghoudwa", r"8odwa", r"lyoum", r"el youm", r"ellila",
    r"christmas", r"easter", r"noel", r"paques", r"ramadan", r"3id", r"aid el",
    r"summer", r"winter", r"spring", r"autumn", r"hiver", r"printemps", r"automne", r"sif", r"chta", r"rbi3", r"5rif",
    r"(next|this|last|coming) (week|month|year)", r"(semaine|mois|annee) (prochaine?|derniere?)", r"cette semaine",
    r"ce mois", r"cet ete", r"l'annee prochaine", r"jem3a jey(a)?", r"chhar jey", r"el 3am ejjey",
    r"\d{1,2}/\d{1,2}", r"\d{1,2}(st|nd|rd|th)\b",
    r"january|february|march|april|june|july|august|september|october|november|december",
    r"janvier|fevrier|avril|juin|juillet|aout|septembre|octobre|novembre|decembre",
    r"monday|tuesday|wednesday|thursday|friday|saturday|sunday",
    r"lundi|mardi|mercredi|jeudi|vendredi|samedi|dimanche",
    r"el tnin|ettleth|larb3a|el khmis|essebt|el a7ad",
]) + r")(?!\w)")

UNLISTED_PLACES = re.compile(r"(?<!\w)(" + "|".join([
    r"tunisia|tunisie|france|italy|italie|italia|turkey|turquie|torkia|spain|espagne|sbanya|germany|allemagne|almania",
    r"england|angleterre|uk|britain|egypt|egypte|masr|morocco|maroc|algeria|algerie|saudi|arabie|s3oudia",
    r"emirates|emirats|qatar|canada|usa|america|amerique|amrika|etats-unis|europe|europa|portugal|belgium|belgique",
    r"switzerland|suisse|netherlands|holland|pays-bas|cairo|caire|nice|kerkennah|carthage|marsa|gammarth",
    r"hammam sousse|port el kantaoui|kantaoui|chott|ksar ghilane|tamerza|sbeitla|dougga|el jem|korba|hergla",
    r"mecca|makkah|mekka|la mecque|medina|antalya|bodrum|tokyo|bangkok|miami|los angeles|toronto",
]) + r")(?!\w)")

ARABIC_SCRIPT = re.compile(r"[؀-ۿ]")


class Matcher:
    def __init__(self, lang):
        self.dates = build_date_index(lang)
        self.places = build_place_index()
        self.date_re = compile_alternation(self.dates)
        self.place_re = compile_alternation(self.places)

    def find_entities(self, text, strict=True):
        """Return (entities, reason). reason is None if the row is clean, else why it should be rejected.
        strict=False skips the unlabeled-date/place checks (e.g. for non-travel rows)."""
        t = norm_text(text)
        taken = [False] * len(t)
        ents = []
        for m in self.date_re.finditer(t):
            s_in, e_in, norm = self.dates[m.group(1)]
            ents.append({"start": m.start() + s_in, "end": m.start() + e_in, "label": "DATE", "norm": norm})
            for i in range(m.start(), m.end()):
                taken[i] = True
        for m in self.place_re.finditer(t):
            if any(taken[m.start():m.end()]):
                continue
            ents.append({"start": m.start(), "end": m.end(), "label": "LOC", "norm": self.places[m.group(1)]})
            for i in range(m.start(), m.end()):
                taken[i] = True

        if strict:
            rest = "".join(" " if taken[i] else c for i, c in enumerate(t))
            if bad := SUSPICIOUS_DATE.search(rest):
                return None, f"unlabeled date: {bad.group(1)}"
            if bad := UNLISTED_PLACES.search(rest):
                return None, f"unlisted place: {bad.group(1)}"
        ents.sort(key=lambda e: e["start"])
        for e in ents:
            e["value"] = text[e["start"]:e["end"]]
        return ents, None
