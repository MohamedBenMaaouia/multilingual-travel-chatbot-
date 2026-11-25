"""Stage 2 - the Supervisor: read the message card + the memory and choose ONE route.

No model, no LLM: plain rules, so every decision is instant and explainable (written in `route_reason`).
The Supervisor is the ONLY writer of the memory keys `topic`, `slots`, `pending` and `conv_lang`.

    1. card -> slot values        ("jarba", role destination -> destination = place = Djerba)
    2. rules, first match wins:
         A  answer to our question   the bot asked for X (pending) and the message contains X
         B  follow-up / correction   topic in memory + new values + no clear new subject
         C  otherwise                filter says OTHER -> Chat agent; else a new question (topic = subcategory)
       New values always overwrite the old ones (a correction is just an overwrite).
    3. doubtful place ("touens")  -> Clarifier: "which place did you mean?"
    4. required info missing      -> Clarifier (max 2 questions per slot, then answer with what we have)
    5. otherwise                  -> the travel expert of the topic's category
"""
import re

from src.labels import SUBCAT_TO_CAT
from src.pipeline.state import ChatState
from src.pipeline.understanding import mask

CONFIDENT = 0.60        # classifier probability above which a DIFFERENT subcategory means "new subject"
LANG_CONFIDENT = 0.80   # fastText probability above which the message's language becomes the conversation's
MAX_ASKS = 2            # questions for the same slot before giving up

EXPERT = {"flights": "flights", "accommodation": "hotels", "destination_info": "destination",
          "budget_money": "money", "local_transport": "local_transport"}

# Minimum information to call the expert's tool. "city" = a place that is a city (our data is per city).
# Everything else has a default in the expert (origin = Tunis, date = today or timetable, nights = 1...).
REQUIRED = {
    "flight_search": ["destination"], "flight_price": ["destination"], "baggage": ["airline"],
    "hotel_search": ["city", "date"], "hotel_price": ["city", "date"], "hotel_facilities": ["city"],
    "weather": ["city"], "activities": ["city"], "visa_documents": ["place"],
    "exchange_rate": ["from_currency"], "trip_cost": ["place"], "payment_methods": ["place"],
    "car_rental": ["place"], "transfer_taxi": ["city"], "public_transport": ["city"],
}


def new_values(state):
    """Step 1: the message card -> slot values (only what this message brings)."""
    v = {}
    for p in state.get("places", []):
        if p["status"] != "ok":
            continue                                   # doubtful or unknown: handled in step 3
        if p["role"] == "origin":
            v["origin"] = p["canonical"]
        else:
            if p["role"] == "destination":
                v["destination"] = p["canonical"]
            v["place"], v["place_kind"] = p["canonical"], p["kind"]
    dates = [d["date"] for d in state.get("dates", []) if d.get("date")]
    if dates:
        v["date"] = dates[0]
    if len(dates) > 1:
        v["checkout"] = dates[1]
    for k in ("nights", "people"):
        if k in state.get("counts", {}):
            v[k] = state["counts"][k]
    cur = state.get("currencies", [])
    if cur:
        v["from_currency"] = cur[0]["code"]
        if cur[0]["amount"] is not None:
            v["amount"] = cur[0]["amount"]
    if len(cur) > 1:
        v["to_currency"] = cur[1]["code"]
    if state.get("airlines"):
        v["airline"] = state["airlines"][0]
    return v


def has(slots, slot):
    """Is this required information present? 'city' = a place that is a city."""
    if slot == "city":
        return slots.get("place_kind") == "city"
    return slot in slots


def answers(pending, values):
    """Rule A: does the message contain what we asked for? (may move a place to origin/destination)"""
    slot = pending["slot"]
    if slot in ("destination", "origin") and slot not in values and "place" in values:
        values[slot] = values["place"]                 # asked "where to?", user answers just "jerba"
        if slot == "origin":
            values.pop("place"), values.pop("place_kind")
    return has(values, slot)


def supervisor_node(state: ChatState):
    slots = dict(state.get("slots") or {})
    topic, pending = state.get("topic"), state.get("pending")
    values = new_values(state)

    # conversation language (fix 4): only a confident detection on a real sentence (2+ words) changes it.
    # fastText is over-confident on content-free messages ("paris", "?"), found in Module 1.
    lang = state.get("lang")
    real_sentence = len(state["text"].split()) >= 2
    if lang and ((state.get("lang_confidence", 0) >= LANG_CONFIDENT and real_sentence) or not state.get("conv_lang")):
        conv_lang = lang
    else:
        conv_lang = state.get("conv_lang") or "en"

    sub, conf = state.get("subcategory"), state.get("sub_confidence", 0.0)
    # The classifier is over-confident on messages that are only a date/place ("<DATE>" -> weather 99%):
    # its opinion counts only if the message has a real word (3+ letters) besides the places and dates.
    has_real_word = any(len(w) >= 3 for w in re.findall(r"\w+", mask(state["text"], state.get("places", []),
                                                                       state.get("dates", [])).replace("<LOC>", "").replace("<DATE>", "")))
    clear_new_subject = sub is not None and conf >= CONFIDENT and sub != topic and has_real_word
    travel = state.get("is_travel") is not False      # None (filter failed) counts as travel

    # ---- step 2: the 3 rules ----
    # Fix 1 (temporary, see docs/open_issues.md #1): A and B do NOT look at the travel filter. It sees the
    # message alone, and continuations like "ghodwa" or "la, ba3d ghodwa" make no sense alone (-> OTHER).
    if pending and answers(pending, values):
        reason = f"A: answer to our question ({pending['slot']})"
        pending = None
    elif topic and values and not clear_new_subject:
        reason = f"B: follow-up/correction (new: {', '.join(sorted(values))})"
    elif not travel:
        return {"route": "chat", "route_reason": "C: travel filter says OTHER", "conv_lang": conv_lang}
    elif conf < CONFIDENT:                          # fix 2: an unsure classifier does not invent a topic
        return {"route": "chat", "route_reason": f"C: unclear message ({sub} only {conf:.0%})",
                "conv_lang": conv_lang}
    else:
        reason = f"C: new question ({sub} {conf:.0%})"
        if sub != topic:
            pending = None                              # new subject: forget the old question
        topic = sub
    if topic in ("flight_search", "flight_price") and "place" in values and "destination" not in values \
            and "origin" not in values:
        values["destination"] = values["place"]         # fix 3: "vol paris" -> a lone place is where they fly to
    slots.update(values)                                # new values overwrite the old ones
    out = {"topic": topic, "slots": slots, "conv_lang": conv_lang}

    # ---- step 3: a doubtful place -> ask which one ----
    airline_words = {a.lower() for a in state.get("airlines", [])}      # "tunisair" is an airline, not "Tunis?"
    doubtful = [p for p in state.get("places", [])
                if p["status"] == "ask" and p["text"].lower() not in airline_words]
    if doubtful:
        p = doubtful[0]
        slot = "origin" if p["role"] == "origin" else "destination" if p["role"] == "destination" else "place"
        return {**out, "route": "clarifier", "route_reason": f"{reason} -> doubtful place '{p['text']}'",
                "pending": {"slot": slot, "times_asked": 1, "doubtful": p["text"]}}

    # ---- step 4: required information missing -> ask (max 2 times per slot) ----
    missing = next((s for s in REQUIRED[topic] if not has(slots, s)), None)
    if missing:
        times = pending["times_asked"] if pending and pending.get("slot") == missing else 0
        if times < MAX_ASKS:
            return {**out, "route": "clarifier", "route_reason": f"{reason} -> missing {missing}",
                    "pending": {"slot": missing, "times_asked": times + 1}}
        reason += f" -> still no {missing} after {MAX_ASKS} questions, answer with what we have"

    # ---- step 5: the expert of the topic's category ----
    route = EXPERT[SUBCAT_TO_CAT[topic]]
    return {**out, "route": route, "route_reason": f"{reason} -> {route}", "pending": None}
