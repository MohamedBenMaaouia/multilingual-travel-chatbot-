"""UNIT TESTS - the 3 nodes that write the reply: Clarifier, Chat agent, Responder.

Mistral is replaced by small fakes (monkeypatch), so each test decides exactly what "Mistral" answers: a good
answer, an answer with an invented price, or a crash. That is how we test the SAFETY parts (numbers check,
fallbacks) without waiting for the real model, and without depending on what it happens to write today.
"""
from types import SimpleNamespace

import pytest

from src.pipeline import chat_agent, responder
from src.pipeline.clarifier import QUESTIONS, clarifier_node
from src.pipeline.responder import allowed_numbers, data_card, numbers_in_reply, prepare_data
from src.pipeline.state import Context
from src.pipeline.supervisor import REQUIRED

from conftest import TODAY

RUNTIME = SimpleNamespace(context=Context(today=TODAY))


@pytest.fixture
def stream_events(monkeypatch):
    """Catch what a node sends to the app (the streamed words and the 'replace' orders) in a list."""
    events = []
    monkeypatch.setattr(chat_agent, "get_stream_writer", lambda: events.append)
    monkeypatch.setattr(responder, "get_stream_writer", lambda: events.append)
    return events


def mistral_says(monkeypatch, module, *pieces, crash=False):
    """Make the fake Mistral of `module` answer with these pieces (or crash like an Ollama failure)."""
    def fake(model, messages, **kwargs):
        fake.messages = messages
        if crash:
            raise ConnectionError("Ollama is down")
        yield from pieces
    monkeypatch.setattr(module, "chat_stream", fake)
    return fake


# ---------- Clarifier ----------

def test_clarifier_asks_in_the_conversation_language():
    out = clarifier_node({"pending": {"slot": "date", "times_asked": 1}, "conv_lang": "tn_latin"})
    assert out["reply"] == "Lel ama nhar?"
    assert out["history"] == [{"role": "bot", "text": "Lel ama nhar?"}]


def test_clarifier_doubtful_place_question():
    out = clarifier_node({"pending": {"slot": "destination", "times_asked": 1, "doubtful": "touens"},
                          "conv_lang": "fr"})
    assert out["reply"] == QUESTIONS["doubtful"]["fr"]


def test_every_required_slot_has_a_question_in_the_3_languages():
    needed = {slot for slots in REQUIRED.values() for slot in slots} | {"doubtful"}
    for slot in needed:
        assert set(QUESTIONS[slot]) == {"en", "fr", "tn_latin"}, slot


# ---------- Chat agent ----------

def test_chat_agent_streams_mistral_reply(monkeypatch, stream_events):
    mistral_says(monkeypatch, chat_agent, "Aslema! ", "Kifech najjem n3awnek?")
    out = chat_agent.chat_agent_node({"text": "salam", "conv_lang": "tn_latin", "lang_confidence": 0.9,
                                      "history": [{"role": "user", "text": "salam"}], "route_reason": ""}, RUNTIME)
    assert out["reply"] == "Aslema! Kifech najjem n3awnek?"
    assert stream_events == [{"token": "Aslema! "}, {"token": "Kifech najjem n3awnek?"}]


def test_chat_agent_falls_back_when_mistral_fails(monkeypatch, stream_events):
    mistral_says(monkeypatch, chat_agent, crash=True)
    out = chat_agent.chat_agent_node({"text": "merci", "conv_lang": "fr", "history": [], "route_reason": ""}, RUNTIME)
    assert out["reply"] == chat_agent.FALLBACK["fr"]
    assert stream_events == [{"replace": chat_agent.FALLBACK["fr"]}]


def test_chat_agent_first_unsure_message_lets_mistral_choose_the_language(monkeypatch, stream_events):
    fake = mistral_says(monkeypatch, chat_agent, "ok")
    chat_agent.chat_agent_node({"text": "salam", "conv_lang": "fr", "lang_confidence": 0.55,
                                "history": [{"role": "user", "text": "salam"}], "route_reason": ""}, RUNTIME)
    assert chat_agent.SAME_LANGUAGE in fake.messages[0]["content"]


def test_chat_agent_unclear_situation_is_told_to_mistral(monkeypatch, stream_events):
    fake = mistral_says(monkeypatch, chat_agent, "ok")
    chat_agent.chat_agent_node({"text": "chnowa?", "conv_lang": "tn_latin", "history": [],
                                "route_reason": "C: unclear message (trip_cost only 40%)"}, RUNTIME)
    assert chat_agent.SITUATIONS["unclear"] in fake.messages[0]["content"]


# ---------- Responder: the numbers check ----------

@pytest.mark.parametrize("reply, numbers", [
    ("Fi Jerba 3la 7aja, 9adech?", set()),               # single digits stuck to letters = Arabizi letters
    ("415 TND", {415.0}),
    ("165dt", {165.0}),                                   # 2+ digits stuck to letters = a price
    ("1,285 TND", {1285.0}),                              # thousands separator
    ("min. 30,000 EUR", {30000.0}),
    ("3.4 dinars", {3.4}),
])
def test_numbers_in_reply(reply, numbers):
    assert numbers_in_reply(reply) == numbers


def test_allowed_numbers_include_data_dates_and_list_sizes():
    allowed = allowed_numbers({"date": "2026-11-27", "flights": [{"price_tnd": 415}, {"price_tnd": 555}]})
    assert {2026.0, 11.0, 27.0, 415.0, 555.0, 2.0} <= allowed


RESULT = {"topic": "flight_search", "tool": "search_flights",
          "arguments": {"origin": "Tunis", "destination": "Paris", "date": "2026-11-27"},
          "defaults_used": {"origin": "Tunis"},
          "data": {"origin": "Tunis", "destination": "Paris", "date": "2026-11-27", "found": True,
                   "flights": [{"airline": "Transavia", "departure": "19:40", "price_tnd": 415}]}}
STATE = {"conv_lang": "en", "result": RESULT, "history": [{"role": "user", "text": "flights to paris friday?"}]}


def test_responder_keeps_a_correct_answer(monkeypatch, stream_events):
    mistral_says(monkeypatch, responder, "Transavia flies to Paris on November 27 at 19:40 for 415 TND.")
    out = responder.responder_node(STATE, RUNTIME)
    assert out["reply_checks"]["numbers_ok"] and out["reply_checks"]["source"] == "mistral"
    assert out["reply_checks"]["language_ok"]


def test_responder_replaces_an_answer_with_an_invented_price(monkeypatch, stream_events):
    mistral_says(monkeypatch, responder, "Transavia flies to Paris for 399 TND.")       # 399 is not in the data
    out = responder.responder_node(STATE, RUNTIME)
    assert out["reply_checks"]["invented_numbers"] == [399.0]
    assert out["reply"] == data_card(prepare_data(RESULT), "en")
    assert stream_events[-1] == {"replace": out["reply"]}                              # the app swaps the text


def test_responder_falls_back_to_the_data_card_when_mistral_fails(monkeypatch, stream_events):
    mistral_says(monkeypatch, responder, crash=True)
    out = responder.responder_node({**STATE, "conv_lang": "tn_latin"}, RUNTIME)
    assert out["reply"].startswith("Hedha elli l9itou:")
    assert out["reply_checks"]["source"] == "fallback (ConnectionError)"


def test_responder_sends_defaults_and_data_to_mistral(monkeypatch, stream_events):
    fake = mistral_says(monkeypatch, responder, "ok")
    responder.responder_node(STATE, RUNTIME)
    request = fake.messages[-1]["content"]
    assert 'DEFAULTS: {"origin": "Tunis"}' in request and '"price_tnd": 415' in request


# ---------- Responder: preparing the data and the data card ----------

def test_prepare_data_keeps_5_hotels_and_drops_facilities_unless_asked():
    hotels = [{"name": f"H{i}", "price_per_night_tnd": 100 + i, "facilities": ["wifi"]} for i in range(8)]
    result = {"topic": "hotel_search", "data": {"found": True, "hotels": hotels}}
    data = prepare_data(result)
    assert len(data["hotels"]) == 5 and "facilities" not in data["hotels"][0]
    assert "facilities" in prepare_data({**result, "topic": "hotel_facilities"})["hotels"][0]
    assert len(result["data"]["hotels"]) == 8                       # the expert's result itself is not changed


def test_data_card_not_found_in_french():
    card = data_card({"found": False, "reason": "no flight on this date"}, "fr")
    assert card == "Désolé, je n'ai rien trouvé :\nraison: no flight on this date"
