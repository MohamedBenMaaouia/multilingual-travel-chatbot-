"""Shared test helpers ("fixtures"). pytest reads this file automatically before the tests.

A fixture is a function that prepares something a test needs; a test asks for it by naming it as an argument:
    def test_x(fake_llm): ...     -> pytest runs fake_llm() first and passes its result in.
"""
from datetime import date

import pytest

# Every test uses the same "today" so dates are reproducible: Wednesday 25 November 2026.
# ghodwa = Thu 26, ba3d ghodwa = Fri 27, nhar el jem3a = Fri 27.
TODAY = date(2026, 11, 25)
MODEL = "mistral:7b-instruct"


@pytest.fixture(scope="session")
def ollama():
    """Skip the test if Mistral can't be reached (Ollama not installed / not running and can't start).
    scope="session": checked once for the whole test run, not once per test."""
    from src.llm.ollama_client import chat
    try:
        chat(MODEL, [{"role": "user", "content": "hi"}], max_tokens=1)
    except Exception as error:
        pytest.skip(f"Ollama / Mistral not reachable: {error}")
    return MODEL


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace Mistral's ANSWER WRITING (Chat agent + Responder) by an instant fake.
    monkeypatch swaps a function for the duration of one test only, then puts the real one back.
    The travel filter is NOT replaced: routing depends on it, and at temperature 0 it is stable.
    Returns the list of prompts the fake received, so a test can look at what Mistral would have been sent."""
    calls = []

    def fake_chat_stream(model, messages, **kwargs):
        calls.append(messages)
        yield "Fake answer from the test."          # no digits: the numbers check always passes

    monkeypatch.setattr("src.pipeline.chat_agent.chat_stream", fake_chat_stream)
    monkeypatch.setattr("src.pipeline.responder.chat_stream", fake_chat_stream)
    return calls
