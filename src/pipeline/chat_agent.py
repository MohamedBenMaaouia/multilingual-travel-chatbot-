"""Stage 3 - the Chat agent: everything that is not a travel request.

Greetings, thanks, off-topic questions, and messages the Supervisor found unclear. Mistral writes a short reply
in the conversation's language; the reply is STREAMED (sent piece by piece to the app while Mistral writes),
so the first words appear after ~1 s even though the whole reply takes ~5-10 s on our GPU.

The fallback is for FAILURES only (Ollama down, an error, or a call stuck for 60 s), never for slowness.
"""
import time

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from src.llm.ollama_client import chat_stream
from src.pipeline.state import ChatState, Context

LANGUAGE_NAMES = {
    "en": "English",
    "fr": "French",
    "tn_latin": "Tunisian Arabic written in Latin letters with digits for Arabic sounds (Arabizi), "
                "never Arabic script, never Modern Standard Arabic",
}

SAME_LANGUAGE = "the same language as the user's message (English, French, or Tunisian Arabizi in Latin letters)"

CHAT_PROMPT = """You are the assistant of a travel agency chatbot. You can help with flights, hotels, local transport,
weather at a destination, things to do, visas, currency exchange, trip budgets and payment.
The user's message is NOT a travel request you can answer: {situation}
Reply in {language}, in one or two short sentences:
- greeting or thanks: answer politely and offer help with their trip;
- off-topic question: say kindly that you can only help with travel, and do NOT answer the question;
- unclear message: ask them to say what they need for their trip.
Never add a translation or anything in brackets."""

# Few-shot: solved examples shown as past turns, so Mistral copies the style (Tunisian replies approved by the user).
FEW_SHOT = [
    ("salam", "Aslema! Ana l'assistant mta3 safar, kifech najjem n3awnek?"),
    ("merci barcha", "Bla jmil! Ken t7eb 7aja okhra 3al safra, ana houni."),
    ("chkoun rba7 el match?", "Samahni, najjem n3awnek ken fel safar: tayarat, hotelet, visa, ta9es..."),
    ("chnowa?", "Ma fhemtekch mli7. Chnowa t7eb ta3ref 3la safrtek?"),
    ("hello", "Hello! I'm your travel assistant. How can I help with your trip?"),
    ("merci !", "Avec plaisir ! Avez-vous une autre question pour votre voyage ?"),
]

SITUATIONS = {
    "unclear": "it is unclear (we could not tell what the user wants).",
    "other": "it is a greeting, thanks, small talk or an off-topic question.",
}

FALLBACK = {
    "en": "I'm your travel assistant. How can I help with your trip?",
    "fr": "Je suis votre assistant de voyage. Comment puis-je vous aider pour votre voyage ?",
    "tn_latin": "Ana l'assistant mta3 safar. Kifech najjem n3awnek fi safrtek?",
}


def chat_agent_node(state: ChatState, runtime: Runtime[Context]):
    t = time.perf_counter()
    lang = state.get("conv_lang") or "en"
    situation = SITUATIONS["unclear" if "unclear" in state.get("route_reason", "") else "other"]
    # fastText is unsure on one-word greetings ("salam" -> fr 55%). On the FIRST message there is no conversation
    # language to fall back on, so we let Mistral reply in the language it sees.
    first_message = len(state.get("history", [])) <= 1
    language = SAME_LANGUAGE if first_message and state.get("lang_confidence", 0) < 0.80 else LANGUAGE_NAMES[lang]
    messages = [{"role": "system", "content": CHAT_PROMPT.format(situation=situation, language=language)}]
    for user_text, bot_text in FEW_SHOT:
        messages += [{"role": "user", "content": user_text}, {"role": "assistant", "content": bot_text}]
    messages.append({"role": "user", "content": state["text"]})
    write = get_stream_writer()                        # sends pieces to whoever is listening (the app)
    pieces = []
    try:
        for piece in chat_stream(runtime.context.llm_model, messages, max_tokens=80):
            pieces.append(piece)
            write({"token": piece})
        reply = "".join(pieces).strip()
        if not reply:
            raise ValueError("empty answer")
    except Exception:                                  # Ollama down / error / stuck 60 s -> never leave the user alone
        reply = FALLBACK[lang]
        write({"replace": reply})                      # tell the app to replace any half-written text
    return {"reply": reply, "history": [{"role": "bot", "text": reply}],
            "timings": {"chat_agent": time.perf_counter() - t}}
