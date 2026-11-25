"""SCENARIO TESTS (integration) - whole conversations through the REAL graph.

Everything is real (fastText, NER + rules, classifier, travel filter with Mistral, Supervisor, experts, memory)
EXCEPT the answer writing (Chat agent + Responder), replaced by an instant fake (fixture `fake_llm` in conftest.py):
these tests check that the bot UNDERSTANDS and ROUTES correctly turn after turn (route, topic, slots, pending,
history), not the wording of the answers. That keeps them stable and takes ~1.5 s per message (the travel filter).

They need Ollama (the travel filter): without it they are skipped.
"""
import pytest

from src.pipeline.graph import build_graph
from src.pipeline.state import Context, new_turn

from conftest import TODAY      # Wednesday 25 November 2026: ghodwa = 26, ba3d ghodwa = nhar el jem3a = 27


class Conversation:
    """One conversation = one thread_id in the graph's memory."""

    def __init__(self, app, name):
        self.app, self.config = app, {"configurable": {"thread_id": name}}

    def say(self, text):
        return self.app.invoke(new_turn(text), self.config, context=Context(today=TODAY))


@pytest.fixture
def app(ollama, fake_llm):
    return build_graph()                 # a new graph = a new, empty memory for every test


@pytest.fixture
def chat(app):
    return Conversation(app, "test")


def test_missing_date_is_asked_then_the_answer_fills_it(chat):
    s = chat.say("nheb hotel f jerba")
    assert s["route"] == "clarifier" and s["reply"] == "Lel ama nhar?"
    assert s["pending"] == {"slot": "date", "times_asked": 1}
    s = chat.say("ghodwa")                                         # alone, the filter says OTHER: rule A ignores it
    assert s["route"] == "hotels" and s["route_reason"].startswith("A:")
    assert s["slots"]["place"] == "Djerba" and s["slots"]["date"] == "2026-11-26"
    assert s["result"]["tool"] == "search_hotels" and s["pending"] is None


def test_flight_without_date_gives_the_timetable_then_a_date_follows_up(chat):
    s = chat.say("nheb billet l paris")
    assert s["route"] == "flights" and "timetable" in s["result"]["data"]
    assert s["result"]["defaults_used"]["origin"] == "Tunis"
    s = chat.say("ghodwa")
    assert s["route"] == "flights" and s["route_reason"].startswith("B:")
    assert s["result"]["arguments"]["date"] == "2026-11-26"


def test_correction_changes_only_the_date(chat):
    chat.say("vol tunis paris ghodwa")
    s = chat.say("la, ba3d ghodwa")
    assert s["route"] == "flights"
    assert s["slots"]["origin"] == "Tunis" and s["slots"]["destination"] == "Paris"
    assert s["slots"]["date"] == "2026-11-27"


def test_follow_up_with_another_city_keeps_the_question(chat):
    s = chat.say("ta9es f jerba ghodwa?")
    assert s["route"] == "destination" and s["topic"] == "weather"
    s = chat.say("w f sousse?")
    assert s["topic"] == "weather" and s["slots"]["place"] == "Sousse"
    assert s["result"]["arguments"] == {"city": "Sousse", "date": "2026-11-26"}


def test_new_subject_replaces_the_old_one(chat):
    chat.say("vol tunis paris ghodwa")
    s = chat.say("chnowa el visa l fransa?")
    assert s["topic"] == "visa_documents" and s["route"] == "destination"
    assert s["result"]["arguments"] == {"destination": "France"}


def test_thanks_go_to_the_chat_agent_and_keep_the_memory(chat):
    chat.say("quel temps fait-il à Djerba demain ?")
    s = chat.say("merci beaucoup")
    assert s["route"] == "chat"
    assert s["topic"] == "weather" and s["slots"]["place"] == "Djerba"      # still remembered for a follow-up
    assert s["conv_lang"] == "fr"


def test_weather_in_a_country_asks_for_the_city(chat):
    s = chat.say("what's the weather like in France?")
    assert s["route"] == "clarifier" and s["reply"] == "Which city?"
    s = chat.say("paris")
    assert s["route"] == "destination" and s["result"]["arguments"]["city"] == "Paris"
    assert s["conv_lang"] == "en"                                           # one word does not change the language


def test_off_topic_goes_to_chat(chat):
    s = chat.say("a3tini recette mlou5iya")
    assert s["route"] == "chat" and s.get("topic") is None


def test_country_word_found_without_the_ner(chat):
    s = chat.say("quels papiers pour un visa pour l'Europe ?")
    assert s["route"] == "destination"
    assert s["result"]["data"]["region"] == "Schengen area"


def test_exchange_rate_with_amount(chat):
    s = chat.say("9adech ysawi 200 euro b dinar?")
    assert s["route"] == "money"
    assert s["result"]["arguments"] == {"from_currency": "EUR", "to_currency": "TND", "amount": 200.0}
    assert s["result"]["data"]["converted"] == 680.0


def test_history_has_one_line_per_message(chat):
    chat.say("nheb hotel f jerba")
    s = chat.say("ghodwa")
    assert [h["role"] for h in s["history"]] == ["user", "bot", "user", "bot"]


def test_two_conversations_have_separate_memories(app):
    a, b = Conversation(app, "user-a"), Conversation(app, "user-b")
    a.say("nheb hotel f jerba")
    s = b.say("ghodwa")                                  # B never talked about a hotel: nothing is waiting
    assert s.get("topic") != "hotel_search" and s["route"] != "hotels"


def test_the_responder_receives_the_conversation_defaults_and_data(chat, fake_llm):
    chat.say("flights to paris on friday?")
    request = fake_llm[-1][-1]["content"]                # last prompt sent to the (fake) Responder
    assert "Conversation: flights to paris on friday?" in request
    assert 'DEFAULTS: {"origin": "Tunis"}' in request and '"destination": "Paris"' in request
