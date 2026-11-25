"""UNIT TESTS - the Supervisor's rules (src/pipeline/supervisor.py).

Each test writes a message card BY HAND (what the Understanding agent would have produced) + the memory, calls
supervisor_node() directly, and checks the route and what it writes in memory. No model, no LLM, no graph:
so a failure here is always a rule problem, never a model mistake.

    rule A   answer to our question      rule B   follow-up / correction      rule C   new question or chat
"""
from src.pipeline.supervisor import supervisor_node


def place(text, message, canonical, kind="city", role="location", status="ok"):
    return {"text": text, "start": message.index(text), "canonical": canonical, "kind": kind,
            "role": role, "status": status}


def date_(text, iso):
    return {"text": text, "date": iso, "precision": "day"}


def card(text, *, places=(), dates=(), sub=None, conf=0.0, is_travel=True, lang="tn_latin", lang_conf=1.0,
         currencies=(), counts=None, airlines=(), **memory):
    """A message card + memory (topic, slots, pending, conv_lang, history...)."""
    return {"text": text, "lang": lang, "lang_confidence": lang_conf, "places": list(places), "dates": list(dates),
            "currencies": list(currencies), "counts": counts or {}, "airlines": list(airlines),
            "subcategory": sub, "sub_confidence": conf, "is_travel": is_travel, **memory}


# ---------- rule C: new questions ----------

def test_new_question_with_everything_goes_to_the_expert():
    t = "ta9es f jerba ghodwa"
    out = supervisor_node(card(t, places=[place("jerba", t, "Djerba")], dates=[date_("ghodwa", "2026-11-26")],
                               sub="weather", conf=0.99))
    assert out["route"] == "destination" and out["topic"] == "weather"
    assert out["slots"] == {"place": "Djerba", "place_kind": "city", "date": "2026-11-26"}
    assert out["route_reason"].startswith("C: new question")


def test_missing_required_information_asks_a_question():
    t = "nheb hotel f jerba"
    out = supervisor_node(card(t, places=[place("jerba", t, "Djerba")], sub="hotel_search", conf=0.99))
    assert out["route"] == "clarifier"
    assert out["pending"] == {"slot": "date", "times_asked": 1}


def test_weather_in_a_country_asks_for_a_city():
    t = "what's the weather like in France?"
    out = supervisor_node(card(t, places=[place("France", t, "France", kind="country")], sub="weather", conf=0.99,
                               lang="en"))
    assert out["route"] == "clarifier" and out["pending"]["slot"] == "city"


def test_visa_works_with_a_country():
    t = "chnowa el visa l fransa?"
    out = supervisor_node(card(t, places=[place("fransa", t, "France", kind="country", role="destination")],
                               sub="visa_documents", conf=0.97))
    assert out["route"] == "destination" and out["slots"]["place"] == "France"


def test_fix3_lone_place_in_a_flight_question_is_the_destination():
    t = "vol paris"
    out = supervisor_node(card(t, places=[place("paris", t, "Paris")], sub="flight_search", conf=0.95, lang="fr"))
    assert out["slots"]["destination"] == "Paris" and out["route"] == "flights"


def test_travel_filter_other_goes_to_chat():
    out = supervisor_node(card("a3tini recette mlou5iya", sub="activities", conf=0.7, is_travel=False))
    assert out["route"] == "chat" and "OTHER" in out["route_reason"]
    assert "topic" not in out                                     # chat does not touch the memory


def test_fix2_unsure_classifier_goes_to_chat_instead_of_inventing_a_topic():
    out = supervisor_node(card("chnowa?", sub="trip_cost", conf=0.40))
    assert out["route"] == "chat" and "unclear" in out["route_reason"]


def test_filter_failure_counts_as_travel():
    t = "ta9es f jerba"
    out = supervisor_node(card(t, places=[place("jerba", t, "Djerba")], sub="weather", conf=0.99, is_travel=None))
    assert out["route"] == "destination"


# ---------- rule A: the answer to our question ----------

def test_rule_A_answer_fills_the_waiting_slot_even_if_the_filter_says_other():
    out = supervisor_node(card("ghodwa", dates=[date_("ghodwa", "2026-11-26")], sub="weather", conf=0.99,
                               is_travel=False, topic="hotel_search",
                               slots={"place": "Djerba", "place_kind": "city"},
                               pending={"slot": "date", "times_asked": 1}))
    assert out["route"] == "hotels" and out["route_reason"].startswith("A:")
    assert out["slots"]["date"] == "2026-11-26" and out["pending"] is None
    assert out["topic"] == "hotel_search"                         # "ghodwa" alone did NOT become a weather question


def test_rule_A_a_lone_place_answers_where_to():
    t = "jerba"
    out = supervisor_node(card(t, places=[place("jerba", t, "Djerba")], sub="weather", conf=0.9,
                               topic="flight_search", slots={}, pending={"slot": "destination", "times_asked": 1}))
    assert out["route"] == "flights" and out["slots"]["destination"] == "Djerba"


# ---------- rule B: follow-ups and corrections ----------

def test_rule_B_correction_overwrites_the_date():
    out = supervisor_node(card("la, ba3d ghodwa", dates=[date_("ba3d ghodwa", "2026-11-27")], sub="flight_search",
                               conf=0.5, is_travel=False, topic="flight_search",
                               slots={"origin": "Tunis", "destination": "Paris", "date": "2026-11-26"}))
    assert out["route"] == "flights" and out["route_reason"].startswith("B:")
    assert out["slots"] == {"origin": "Tunis", "destination": "Paris", "date": "2026-11-27"}


def test_rule_B_follow_up_with_a_new_city_keeps_the_topic():
    t = "w f sousse?"
    out = supervisor_node(card(t, places=[place("sousse", t, "Sousse")], sub="weather", conf=0.8,
                               topic="weather", slots={"place": "Djerba", "place_kind": "city", "date": "2026-11-26"}))
    assert out["route"] == "destination" and out["slots"]["place"] == "Sousse"
    assert out["slots"]["date"] == "2026-11-26"                   # kept from memory


def test_a_date_alone_is_not_a_new_subject_even_if_the_classifier_is_sure():
    out = supervisor_node(card("ghodwa", dates=[date_("ghodwa", "2026-11-26")], sub="weather", conf=0.99,
                               topic="flight_search", slots={"destination": "Paris"}))
    assert out["topic"] == "flight_search" and out["route"] == "flights"


def test_a_clear_new_subject_wins_over_the_follow_up():
    t = "chnowa el visa l fransa?"
    out = supervisor_node(card(t, places=[place("fransa", t, "France", kind="country", role="destination")],
                               sub="visa_documents", conf=0.97, topic="flight_search",
                               slots={"origin": "Tunis", "destination": "Paris"}))
    assert out["topic"] == "visa_documents" and out["route_reason"].startswith("C:")


def test_thanks_after_an_answer_goes_to_chat():
    out = supervisor_node(card("merci beaucoup", sub="hotel_price", conf=0.5, is_travel=False, lang="fr",
                               topic="weather", slots={"place": "Djerba", "place_kind": "city"}))
    assert out["route"] == "chat"


# ---------- doubtful places and the question limit ----------

def test_doubtful_place_asks_which_place():
    t = "nheb nemchi l touens"
    out = supervisor_node(card(t, places=[place("touens", t, "Tunis", role="destination", status="ask")],
                               sub="flight_search", conf=0.9))
    assert out["route"] == "clarifier"
    assert out["pending"] == {"slot": "destination", "times_asked": 1, "doubtful": "touens"}


def test_an_airline_name_is_not_a_doubtful_place():
    t = "baggage tunisair"
    out = supervisor_node(card(t, places=[place("tunisair", t, "Tunis", status="ask")], airlines=["Tunisair"],
                               sub="baggage", conf=0.99, lang="en"))
    assert out["route"] == "flights" and out["slots"]["airline"] == "Tunisair"


def test_after_two_questions_the_bot_answers_with_what_it_has():
    out = supervisor_node(card("ma na3rafch", sub="hotel_search", conf=0.9, topic="hotel_search",
                               slots={"place": "Djerba", "place_kind": "city"},
                               pending={"slot": "date", "times_asked": 2}))
    assert out["route"] == "hotels" and "still no date" in out["route_reason"]
    assert out["pending"] is None


# ---------- the conversation language ----------

def test_a_confident_sentence_changes_the_conversation_language():
    out = supervisor_node(card("je veux un vol pour paris", lang="fr", lang_conf=0.99, sub="flight_search",
                               conf=0.9, conv_lang="en"))
    assert out["conv_lang"] == "fr"


def test_fix4_a_one_word_message_does_not_change_the_language():
    t = "paris"
    out = supervisor_node(card(t, places=[place("paris", t, "Paris")], lang="tn_latin", lang_conf=0.98,
                               conv_lang="fr", topic="weather", slots={}, pending={"slot": "city", "times_asked": 1}))
    assert out["conv_lang"] == "fr"


def test_the_first_message_sets_the_language_even_if_unsure():
    out = supervisor_node(card("salam", lang="fr", lang_conf=0.55, is_travel=False))
    assert out["conv_lang"] == "fr"
