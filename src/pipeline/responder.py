"""Stage 4 - the Responder: turn the expert's `result` into the reply, in the conversation's language.

    1. prepare the data   keep what answers the question (5 cheapest hotels, facilities only if asked...)
    2. build the prompt   instructions + few-shot examples + the conversation + DEFAULTS + DATA (as JSON text)
    3. Mistral writes     streamed (the app shows the words while they are written)
    4. check              numbers: every number in the reply must exist in the data -> otherwise REPLACE (user's choice)
                          language: fastText on the reply -> only FLAGGED in reply_checks (user's choice)
    5. fallback           a generic "data card" (the key facts as short lines) if Mistral fails or invents a number
"""
import copy
import json
import re
import time

from langgraph.config import get_stream_writer
from langgraph.runtime import Runtime

from src.lang_detect.detect import detect_language
from src.llm.ollama_client import chat_stream
from src.pipeline.chat_agent import LANGUAGE_NAMES
from src.pipeline.state import ChatState, Context

SYSTEM_PROMPT = """You are the assistant of a travel agency. Answer the user's request in {language}.
Use ONLY the facts in DATA. Never invent a number, a name or a time. Prices are in TND (Tunisian dinars).
If DATA says "found": false, explain the reason simply and offer what DATA suggests (other dates, the cheapest price...).
If DEFAULTS is not empty, say what you assumed (for example "departing from Tunis").
Climate data is a monthly average, not a forecast for that exact day.
Never add a translation or anything in brackets."""

# Solved examples (DATA -> answer). Tunisian answers to be validated by the user.
FEW_SHOT = [
    ("Conversation: flights to paris friday\nDEFAULTS: {\"origin\": \"Tunis\"}\n"
     "DATA: {\"origin\": \"Tunis\", \"destination\": \"Paris\", \"date\": \"2026-11-27\", \"found\": true, \"flights\": ["
     "{\"airline\": \"Transavia\", \"departure\": \"19:40\", \"arrival\": \"22:05\", \"price_tnd\": 415, \"checked_bag_kg\": 0}, "
     "{\"airline\": \"Tunisair\", \"departure\": \"19:45\", \"arrival\": \"22:10\", \"price_tnd\": 555, \"checked_bag_kg\": 23}]}",
     "Departing from Tunis, there are 2 flights to Paris on Friday, November 27: Transavia at 19:40, arriving at 22:05, "
     "for 415 TND without a checked bag, and Tunisair at 19:45, arriving at 22:10, for 555 TND with a 23 kg bag included."),
    ("Conversation: hotel a tozeur pas cher, moins de 50 dinars\nDEFAULTS: {\"nights\": 1}\n"
     "DATA: {\"city\": \"Tozeur\", \"checkin\": \"2026-12-01\", \"checkout\": \"2026-12-02\", \"nights\": 1, \"found\": false, "
     "\"reason\": \"no hotel under 50 TND per night\", \"cheapest_tnd\": 145}",
     "Je n'ai trouvé aucun hôtel à Tozeur à moins de 50 TND la nuit pour le 1er décembre : le moins cher commence à "
     "145 TND la nuit. J'ai compté une seule nuit, dites-moi si vous restez plus longtemps."),
    ("Conversation: ta9es f jerba fi juillet?\nDEFAULTS: {}\n"
     "DATA: {\"found\": true, \"city\": \"Djerba\", \"date\": \"2026-07-15\", \"temp_min\": 24, \"temp_max\": 33, "
     "\"condition\": \"sunny\", \"rain_days\": 0, \"sea_temp\": 26, \"note\": \"monthly climate average, not a day forecast\"}",
     "Fi Jerba fi juillet el jaw s5oun w mchammes: bin 24 w 33 daraja, w el bhar 26 daraja, w ma fammech mtar. "
     "Hedha mo3adel mta3 el chhar, mouch météo mta3 nhar bidou."),
    ("Conversation: fama tayara men jerba l brussel nhar el jem3a?\nDEFAULTS: {}\n"
     "DATA: {\"origin\": \"Djerba\", \"destination\": \"Brussels\", \"date\": \"2026-11-27\", \"found\": false, "
     "\"reason\": \"this route only flies in summer (April to October)\"}",
     "Ma fammech tayarat bin Jerba w Brussel nhar el jem3a: el ligne hedhi temchi ken fel sif, men avril l octobre."),
]


# ---------- 1. prepare the data ----------

def prepare_data(result):
    data = copy.deepcopy(result["data"])
    if "hotels" in data:
        data["hotels"] = data["hotels"][:5]                       # the 5 cheapest are enough to answer
        if result["topic"] != "hotel_facilities":
            for h in data["hotels"]:
                h.pop("facilities", None)                         # long lists, only useful for facility questions
    if "flights" in data:
        data["flights"] = data["flights"][:5]
    return data


# ---------- 4. the numbers check ----------

# A number in the reply: 2+ digits may touch letters ("165dt"), but a SINGLE digit touching letters is an Arabizi
# letter ("3la", "7aja", "9adech") and is ignored. "1 285" / "1,285" are read as 1285.
REPLY_NUMBER = re.compile(r"(?<![A-Za-z\d])(\d{1,3}(?:[   ,.]\d{3})+|\d+(?:[.,]\d+)?)(?![A-Za-z]|\d)"
                          r"|(?<![A-Za-z\d])(\d{2,})(?=[A-Za-z])")


def _as_number(text):
    text = text.replace(" ", " ").replace(" ", " ")
    if re.fullmatch(r"\d{1,3}(?:[ ,.]\d{3})+", text):              # thousands separators
        return float(re.sub(r"[ ,.]", "", text))
    return float(text.replace(",", "."))


def numbers_in_reply(reply):
    return {_as_number(a or b) for a, b in REPLY_NUMBER.findall(reply)}


def allowed_numbers(*sources):
    """Every number written anywhere in the data / arguments / defaults / the user's messages, + list sizes."""
    text = " ".join(json.dumps(s, ensure_ascii=False) if not isinstance(s, str) else s for s in sources)
    # same reader as the reply ("30,000" = 30000) + plain numbers ("2026-11-27" = 2026, 11, 27)
    allowed = numbers_in_reply(text) | {float(n) for n in re.findall(r"\d+(?:\.\d+)?", text)}

    def sizes(x):
        if isinstance(x, list):
            allowed.add(float(len(x)))
            for y in x:
                sizes(y)
        elif isinstance(x, dict):
            for y in x.values():
                sizes(y)
    for s in sources:
        sizes(s)
    return allowed | {float(n) for n in range(0, 11)}            # 0-10: counts like "2 flights", "1 night"


# ---------- 5. the fallback: a generic data card ----------

LABELS = {   # a few common keys translated; other keys are shown as they are
    "price_tnd": {"en": "price", "fr": "prix", "tn_latin": "soum"},
    "price_per_night_tnd": {"en": "per night", "fr": "par nuit", "tn_latin": "lel lila"},
    "total_tnd": {"en": "total", "fr": "total", "tn_latin": "el kol"},
    "departure": {"en": "departure", "fr": "départ", "tn_latin": "te5rej"},
    "arrival": {"en": "arrival", "fr": "arrivée", "tn_latin": "tousel"},
    "temp_min": {"en": "min", "fr": "min", "tn_latin": "a9al"},
    "temp_max": {"en": "max", "fr": "max", "tn_latin": "akther"},
    "reason": {"en": "reason", "fr": "raison", "tn_latin": "sbeb"},
}
CARD_TITLE = {
    True: {"en": "Here is what I found:", "fr": "Voici ce que j'ai trouvé :", "tn_latin": "Hedha elli l9itou:"},
    False: {"en": "Sorry, I found nothing:", "fr": "Désolé, je n'ai rien trouvé :", "tn_latin": "Samahni, ma l9it chay:"},
}


def data_card(data, lang):
    def label(k):
        return LABELS.get(k, {}).get(lang, k.replace("_tnd", " (TND)").replace("_", " "))

    def line(d):
        parts = []
        for k, v in d.items():
            if isinstance(v, list):
                v = ", ".join(str(x) for x in v)
            if v not in (None, "", []) and not isinstance(v, dict):
                parts.append(f"{label(k)}: {v}")
        return " · ".join(parts)

    lines = [CARD_TITLE[bool(data.get("found"))][lang]]
    lists = [v for v in data.values() if isinstance(v, list) and v and isinstance(v[0], dict)]
    lines.append(line({k: v for k, v in data.items() if k != "found" and not (isinstance(v, list) and v and isinstance(v[0], dict))}))
    for items in lists:
        lines += ["- " + line(item) for item in items]
    return "\n".join(l for l in lines if l)


# ---------- the node ----------

def responder_node(state: ChatState, runtime: Runtime[Context]):
    t = time.perf_counter()
    lang = state.get("conv_lang") or "en"
    result = state["result"]
    data = prepare_data(result)
    users = [h["text"] for h in state.get("history", []) if h["role"] == "user"][-3:]     # the request may span turns
    request = (f"Conversation: {' / '.join(users)}\nDEFAULTS: {json.dumps(result['defaults_used'], ensure_ascii=False)}\n"
               f"DATA: {json.dumps(data, ensure_ascii=False)}")

    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(language=LANGUAGE_NAMES[lang])}]
    for example, answer in FEW_SHOT:
        messages += [{"role": "user", "content": example}, {"role": "assistant", "content": answer}]
    messages.append({"role": "user", "content": request})

    write = get_stream_writer()
    pieces, checks = [], {}
    try:
        # long answers are fine (user's choice): 600 tokens is only a safety cap; 150 s = really stuck
        for piece in chat_stream(runtime.context.llm_model, messages, max_tokens=600, total_timeout=150):
            pieces.append(piece)
            write({"token": piece})
        reply = "".join(pieces).strip()
        if not reply:
            raise ValueError("empty answer")
        invented = numbers_in_reply(reply) - allowed_numbers(data, result["arguments"], result["defaults_used"], users)
        checks["numbers_ok"] = not invented
        checks["invented_numbers"] = sorted(invented)
        detected, conf = detect_language(reply)
        checks["language_ok"] = detected == lang
        checks["reply_language"] = f"{detected} ({conf:.0%})"
        if invented:                                              # 1a: never leave a wrong number on screen
            reply = data_card(data, lang)
            write({"replace": reply})
            checks["source"] = "fallback (invented numbers)"
        else:
            checks["source"] = "mistral"
    except Exception as error:                                    # Ollama down / error / stuck 150 s
        reply = data_card(data, lang)
        write({"replace": reply})
        checks["source"] = f"fallback ({type(error).__name__})"
    return {"reply": reply, "reply_checks": checks, "history": [{"role": "bot", "text": reply}],
            "timings": {"responder": time.perf_counter() - t}}
