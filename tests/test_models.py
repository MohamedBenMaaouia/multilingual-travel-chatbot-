"""MODEL TESTS - do the 3 saved models load and behave on CLEAR examples?

    fastText (models/lang_detect/), XLM-R NER (models/ner_xlmr/), MLP classifier (models/classifier/)

These do NOT measure accuracy (that is done in the notebooks, on val and test). They catch a broken setup:
a missing or wrong model file, a changed library version, a preprocessing change that silently breaks the input.
The models are large files kept out of git, so if one is missing the test is skipped with the reason.
Loading takes ~15 s the first time (the NER model is 546 MB), then each prediction is fast.
"""
from pathlib import Path

import pytest

MODELS = Path(__file__).resolve().parents[1] / "models"


def need(path):
    if not (MODELS / path).exists():
        pytest.skip(f"model not found: models/{path} (re-create it with its notebook)")


# ---------- Module 1: language detection (fastText) ----------

@pytest.mark.parametrize("text, lang", [
    ("nheb nemchi l jerba ghodwa, 9adech el billet?", "tn_latin"),
    ("je voudrais réserver un hôtel à paris pour trois nuits", "fr"),
    ("i would like to book a flight to paris tomorrow", "en"),
])
def test_language_detection(text, lang):
    need("lang_detect/fasttext_lang.bin")
    from src.lang_detect.detect import detect_language
    got, confidence = detect_language(text)
    assert got == lang and 0.8 <= confidence <= 1.0


def test_language_detection_empty_text():
    need("lang_detect/fasttext_lang.bin")
    from src.lang_detect.detect import detect_language
    assert detect_language("   ") == (None, 0.0)


# ---------- Module 4: NER (XLM-R) ----------

@pytest.mark.parametrize("text, expected", [
    ("nheb billet men tunis l jerba ghodwa", [("tunis", "LOC"), ("jerba", "LOC"), ("ghodwa", "DATE")]),
    ("quel temps fait-il à Djerba demain ?", [("Djerba", "LOC"), ("demain", "DATE")]),
    ("hotels in sousse next weekend", [("sousse", "LOC"), ("next weekend", "DATE")]),
])
def test_ner_finds_places_and_dates(text, expected):
    need("ner_xlmr")
    from src.ner.extract import extract_entities
    found = extract_entities(text)
    assert [(e["text"], e["label"]) for e in found] == expected
    assert all(text[e["start"]:e["end"]] == e["text"] for e in found)        # offsets point at the right text


# ---------- Module 3: subcategory classifier (MLP on TF-IDF) ----------

@pytest.mark.parametrize("masked, subcategory", [
    ("9adech el billet l <LOC> <DATE> ?", "flight_price"),
    ("nheb hotel f <LOC>", "hotel_search"),
    ("what's the weather like in <LOC> <DATE>?", "weather"),
    ("do i need a visa for <LOC>?", "visa_documents"),
    ("how many kg of luggage can i take with tunisair?", "baggage"),
    ("est-ce qu'il y a le wifi dans l'hôtel à <LOC> ?", "hotel_facilities"),
    ("how do i get from the airport to <LOC> by taxi?", "transfer_taxi"),
])
def test_classifier_on_clear_questions(masked, subcategory):
    need("classifier/mlp_tfidf.joblib")
    from src.classifier.classify import classify
    c = classify(masked)
    assert c["subcategory"] == subcategory and c["confidence"] >= 0.9
    assert c["top2"][0][0] == subcategory and len(c["top2"]) == 2


def test_classifier_category_comes_from_the_subcategory():
    need("classifier/mlp_tfidf.joblib")
    from src.classifier.classify import classify
    from src.labels import SUBCAT_TO_CAT
    c = classify("how much is 200 euros in dinars?")
    assert c["category"] == SUBCAT_TO_CAT[c["subcategory"]]


# ---------- Module 2: travel filter (Mistral, needs Ollama) ----------

@pytest.mark.parametrize("text, is_travel", [
    ("i need a flight from tunis to paris next friday", True),
    ("9adech el hotel f jerba lila?", True),
    ("chkoun rba7 el match lbare7?", False),
    ("merci beaucoup", False),
])
def test_travel_filter_on_clear_messages(ollama, text, is_travel):
    from src.travel_filter.llm_filter import classify_travel
    assert classify_travel(text, model=ollama, version=3)["is_travel"] is is_travel
