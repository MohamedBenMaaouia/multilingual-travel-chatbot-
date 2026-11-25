"""UNIT TESTS - the rules around the NER model (src/ner/rules.py).

The NER model says WHERE the places and dates are; these rules say WHAT they mean (canonical place, ISO date...).
The NER model is NOT used here: when a test needs entities, it writes them by hand, so the file runs in seconds
and a failure points at the rules, not at the model.
"""
import pytest

from src.ner.rules import (extract_slots, find_airlines, find_counts, find_country_words, find_currencies,
                           normalize_date, normalize_place)

from conftest import TODAY      # Wednesday 25 November 2026


# ---------- places ----------

@pytest.mark.parametrize("text, canonical, kind", [         # parametrize = the same test run once per line
    ("jarba", "Djerba", "city"),
    ("Djerba", "Djerba", "city"),
    ("jerba", "Djerba", "city"),
    ("tunis", "Tunis", "city"),
    ("fransa", "France", "country"),
    ("france", "France", "country"),
    ("orouba", "Schengen area", "region"),
    ("l'europe", "Schengen area", "region"),                  # the article is removed
    ("el khalij", "Gulf", "region"),
])
def test_known_place_spellings(text, canonical, kind):
    p = normalize_place(text)
    assert (p["canonical"], p["kind"], p["status"]) == (canonical, kind, "ok")


@pytest.mark.parametrize("text, guess", [("frnace", "France"), ("touens", "Tunis")])
def test_close_typo_is_asked_not_guessed(text, guess):
    p = normalize_place(text)                                  # similarity 0.80-0.85: ask the user
    assert p["status"] == "ask" and p["canonical"] == guess


def test_la_marsa_is_not_confused_with_marseille():
    assert normalize_place("la marsa")["status"] == "unknown"   # 0.77 < 0.80: not even asked


def test_unknown_place():
    assert normalize_place("xyzabc") == {"canonical": None, "kind": None, "score": 0.38, "status": "unknown"}


def test_country_safety_net_finds_country_and_region_words():
    assert find_country_words("visa process to europe")[0]["canonical"] == "Schengen area"
    assert find_country_words("chnowa el visa l torkia")[0]["canonical"] == "Turkey"


def test_country_safety_net_skips_airline_names():
    assert find_country_words("flying air france to paris") == []
    assert find_country_words("qatar airways") == []


# ---------- dates (today = Wednesday 25 November 2026) ----------

@pytest.mark.parametrize("text, iso, how", [
    ("ghodwa", "2026-11-26", "lexicon"),
    ("ba3d ghodwa", "2026-11-27", "lexicon"),
    ("demain matin", "2026-11-26", "lexicon"),               # time of day is dropped
    ("tomorrow", "2026-11-26", "lexicon"),
    ("nhar el jem3a", "2026-11-27", "lexicon"),               # next Friday
    ("ba3d 3 iyem", "2026-11-28", "rule"),
    ("dans trois jours", "2026-11-28", "rule"),
    ("fin avril", "2027-04-25", "rule"),                      # April has passed this year -> next year
    ("a5er avril", "2027-04-25", "rule"),
    ("christmas", "2026-12-25", "rule"),
    ("le 5", "2026-12-05", "rule"),                           # the 5th has passed this month -> next month
    ("nhar 30", "2026-11-30", "rule"),
    ("this week", "2026-11-25", "rule"),
    ("ghodwaa", "2026-11-26", "fuzzy"),                       # typo
])
def test_dates_become_iso(text, iso, how):
    d = normalize_date(text, TODAY)
    assert (d["date"], d["how"]) == (iso, how)


@pytest.mark.parametrize("text", ["ramadan", "blabla"])
def test_unknown_dates_are_not_guessed(text):
    assert normalize_date(text, TODAY) is None                # Ramadan moves every year: the bot asks


# ---------- currencies, counts, airlines ----------

def test_exchange_question_gives_from_and_to_currency_with_amount():
    cur = find_currencies("9adech ysawi 200 euro b dinar?")
    assert [(c["code"], c["amount"]) for c in cur] == [("EUR", 200.0), ("TND", None)]


def test_currency_typo_and_multi_word_name():
    cur = find_currencies("Je veux changer 500 eurooo en dinars tunisiens")
    assert [c["code"] for c in cur] == ["EUR", "TND"] and cur[0]["amount"] == 500.0


def test_ambiguous_dollar_follows_the_country_of_the_place():
    assert find_currencies("100 dollars")[0]["code"] == "USD"
    assert find_currencies("hotel in Montreal around 200 dollars", ["Canada"])[0]["code"] == "CAD"


def test_amount_written_in_words():
    assert find_currencies("zouz dinar")[0]["amount"] == 2


@pytest.mark.parametrize("text, counts", [
    ("3 lyeli", {"nights": 3}),
    ("zouz nfar", {"people": 2}),
    ("2 nfar w 5 nights", {"people": 2, "nights": 5}),
    ("my bag is 25 kg", {"kg": 25}),
    ("a7na 4 fel safra", {"people": 4}),
    ("a7na 4", {"people": 4}),                                # the number ends the message (bug fixed)
    ("on est 3", {"people": 3}),                              # two-word phrases (bug fixed)
    ("we are three, hotel in sousse", {"people": 3}),
])
def test_counts(text, counts):
    assert find_counts(text) == counts


def test_airlines_with_typo_and_two_airlines():
    assert find_airlines("baggage allowance with tunisaire") == ["Tunisair"]
    assert find_airlines("nouvelair w tunisair") == ["Nouvelair", "Tunisair"]


# ---------- everything together, with hand-written NER entities ----------

def entities(text, locs=(), dates=()):
    """Build the NER output by hand: [{text, label, start, end}]."""
    out = []
    for word, label in [(w, "LOC") for w in locs] + [(w, "DATE") for w in dates]:
        start = text.index(word)
        out.append({"text": word, "label": label, "start": start, "end": start + len(word)})
    return out


def test_extract_slots_full_tunisian_request():
    text = "nheb billet men tunis l jarba ba3d ghodwa l 2 nfar b 300 dt"
    s = extract_slots(text, TODAY, entities(text, locs=["tunis", "jarba"], dates=["ba3d ghodwa"]))
    assert [(p["canonical"], p["role"]) for p in s["places"]] == [("Tunis", "origin"), ("Djerba", "destination")]
    assert s["dates"][0]["date"] == "2026-11-27"
    assert s["counts"] == {"people": 2}
    assert [(c["code"], c["amount"]) for c in s["currencies"]] == [("TND", 300.0)]
    assert s["to_confirm"] == []


def test_extract_slots_keyword_style_two_places_origin_then_destination():
    text = "vol tunis paris"
    s = extract_slots(text, TODAY, entities(text, locs=["tunis", "paris"]))
    assert [p["role"] for p in s["places"]] == ["origin", "destination"]


def test_extract_slots_country_found_by_the_safety_net_when_ner_finds_nothing():
    s = extract_slots("visa process to europe", TODAY, entities=[])
    assert s["places"][0]["canonical"] == "Schengen area" and s["places"][0]["kind"] == "region"


def test_extract_slots_relative_date_found_by_the_safety_net():
    s = extract_slots("fama bac ba3d 3 iyem?", TODAY, entities=[])
    assert s["dates"][0]["date"] == "2026-11-28"


def test_extract_slots_doubtful_place_goes_to_to_confirm():
    text = "nheb nemchi l touens"
    s = extract_slots(text, TODAY, entities(text, locs=["touens"]))
    assert s["to_confirm"] == ["touens"]
