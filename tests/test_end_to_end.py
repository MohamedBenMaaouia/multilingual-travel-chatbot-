"""END-TO-END TESTS (slow) - the real bot, with Mistral writing the answers, through the entry point the UI uses
(src/pipeline/chatbot.py). Several minutes (Mistral writes ~8 tokens/s on this laptop), so they only run on
demand:   pytest -m slow

What they check is what must ALWAYS hold, whatever Mistral writes:
    - a reply always comes back (never empty, never a crash)
    - no invented number stays on screen (either Mistral passed the numbers check, or the data card replaced it)
    - streaming works: words arrive before the end
    - if Mistral is unreachable, the fallbacks answer instead
Answer QUALITY is not tested here: that is the 30-query human evaluation.
"""
import pytest

from src.pipeline.chatbot import handle_query, stream_query, warm_up
from src.pipeline.graph import build_graph
from src.pipeline.state import Context, new_turn

from conftest import TODAY

pytestmark = pytest.mark.slow          # every test in this file is "slow"


def check_safe(reply, debug):
    assert reply.strip()
    checks = debug["reply_checks"]
    if checks["source"] == "mistral":
        assert checks["numbers_ok"]                  # Mistral's text was kept only because every number is real
    else:
        assert checks["source"].startswith("fallback")


def test_warm_up_loads_everything(ollama):
    assert warm_up() > 0


@pytest.mark.parametrize("message", [
    "flights to paris on friday?",
    "quel temps fait-il à Djerba demain ?",
    "9adech ysawi 200 euro b dinar?",
])
def test_real_answers_are_safe(ollama, message):
    reply, debug = handle_query(message, thread_id=f"e2e {message}", today=TODAY)
    assert debug["route"] in ("flights", "destination", "money")
    check_safe(reply, debug)


def test_english_answer_is_in_english(ollama):
    reply, debug = handle_query("how much is a hotel in Sousse for 2 nights from friday?", thread_id="e2e en",
                                today=TODAY)
    check_safe(reply, debug)
    assert debug["reply_checks"]["language_ok"]


def test_streaming_sends_words_before_the_end(ollama):
    kinds = [kind for kind, _ in stream_query("hello!", thread_id="e2e stream", today=TODAY)]
    assert kinds[0] == "token" and kinds[-1] == "done"


def test_clarification_and_answer_through_the_entry_point(ollama):
    reply, debug = handle_query("nheb hotel f jerba", thread_id="e2e two turns", today=TODAY)
    assert reply == "Lel ama nhar?" and debug["route"] == "clarifier"
    reply, debug = handle_query("nhar el jem3a, 3 lyeli", thread_id="e2e two turns", today=TODAY)
    assert debug["route"] == "hotels" and debug["memory"]["slots"]["nights"] == 3
    check_safe(reply, debug)


def test_fallbacks_when_mistral_is_unreachable():
    """A model name that does not exist = every Mistral call fails: the bot must still answer."""
    app = build_graph()
    ctx = Context(today=TODAY, llm_model="no-such-model:latest")
    s = app.invoke(new_turn("flights to paris on friday?"), {"configurable": {"thread_id": "down"}}, context=ctx)
    assert s["is_travel"] is None                                  # filter failed -> counted as travel
    assert s["route"] == "flights"
    assert s["reply_checks"]["source"].startswith("fallback")
    assert s["reply"].startswith("Here is what I found:")          # the data card
