"""The state of the chatbot graph: the one shared "notebook" every agent reads and writes.

Rules (see docs/agentic_architecture.md):
- A node receives the whole state and returns ONLY the keys it changed; LangGraph merges them in.
- Every key has ONE owner (the node that writes it), except where a reducer says how to merge.
- Two lifetimes:
    THIS MESSAGE  - rewritten every turn (reset by new_turn())
    MEMORY        - kept between turns by the checkpointer (one conversation = one thread_id)
- Fixed settings for the whole run (today's date, LLM model) are NOT state: they are the Context.
"""
import operator
from dataclasses import dataclass, field
from datetime import date
from typing import Annotated, Literal, Optional, TypedDict

Lang = Literal["en", "fr", "tn_latin"]

# Which agent the Supervisor sends the message to (stage 2)
Route = Literal["flights", "hotels", "destination", "money", "local_transport",
                "clarifier", "chat"]


def merge_timings(old, new):
    """Reducer for `timings`: parallel nodes each add their own entry.
    new = None means "start a new turn": forget the previous timings."""
    if new is None:
        return {}
    return {**(old or {}), **new}


# ---------- small typed pieces (documentation for the editor; not checked at runtime) ----------

class Place(TypedDict):
    text: str                  # as the user wrote it: "jarba"
    canonical: Optional[str]   # our name: "Djerba" (None = unknown)
    kind: Optional[str]        # city | country | region
    role: str                  # origin | destination | location
    status: str                # ok | ask | unknown


class DateSlot(TypedDict):
    text: str                  # "ba3d ghodwa"
    date: Optional[str]        # "2026-11-26" (None = not understood)
    precision: str             # day | week | weekend | month | season | period


class Slots(TypedDict, total=False):
    """Information collected over the conversation (MEMORY). Only the Supervisor writes it."""
    origin: str
    destination: str
    place: str                 # a city / country / region for weather, visa, activities...
    place_kind: str            # city | country | region (weather and hotels need a city)
    date: str                  # ISO date (travel date, check-in...)
    checkout: str
    nights: int
    people: int
    amount: float
    from_currency: str
    to_currency: str
    budget: float
    airline: str


class Pending(TypedDict, total=False):
    """A question the bot asked and is waiting for (MEMORY). Written only by the Supervisor."""
    slot: str                  # what we asked for: "date", "destination", "city", ...
    times_asked: int           # the Supervisor stops asking after 2
    doubtful: str              # the unclear text, when we ask "which place did you mean by 'touens'?"


class ChatState(TypedDict, total=False):
    # ===== THIS MESSAGE (reset by new_turn) =====
    text: str                                  # the user's message

    # -- message card, stage 1 (each part written by its own Understanding node) --
    lang: Optional[Lang]                       # fastText (Module 1)
    lang_confidence: float
    places: list[Place]                        # XLM-R NER + rules (Module 4)
    dates: list[DateSlot]
    currencies: list[dict]                     # [{"code", "text", "amount"}]
    counts: dict                               # {"nights": 3, "people": 2, ...}
    airlines: list[str]
    to_confirm: list[str]                      # doubtful values to confirm with the user
    subcategory: Optional[str]                 # MLP classifier (Module 3)
    category: Optional[str]
    sub_confidence: float
    top2: list                                 # [(subcategory, prob), (subcategory, prob)]
    is_travel: Optional[bool]                  # travel filter, Mistral (Module 2, unchanged)

    # -- stages 2-4 --
    route: Optional[Route]                     # Supervisor
    route_reason: str                          # which Supervisor rule decided and why (debug panel, report)
    result: Optional[dict]                     # the expert's structured result (from the data tools)
    reply: Optional[str]                       # Responder: the final answer
    reply_checks: dict                         # Responder: numbers check + language check (debug panel)
    timings: Annotated[dict, merge_timings]    # seconds per node, for the debug panel

    # ===== MEMORY (kept between turns) =====
    conv_lang: Optional[Lang]                  # the conversation's language (for messages like "paris ?")
    topic: Optional[str]                       # current subcategory (for follow-ups: "w f sousse?")
    slots: Slots                               # collected information
    pending: Optional[Pending]                 # question waiting for an answer
    last_result: Optional[dict]                # previous expert result ("fama ar5es?" = anything cheaper?)
    history: Annotated[list[dict], operator.add]   # [{"role": "user"|"bot", "text": ...}], appended


def new_turn(text):
    """Input for one user message: the new text + every THIS-MESSAGE field reset.
    MEMORY fields are not listed, so the checkpointer keeps them."""
    return {
        "text": text,
        "lang": None, "lang_confidence": 0.0,
        "places": [], "dates": [], "currencies": [], "counts": {}, "airlines": [], "to_confirm": [],
        "subcategory": None, "category": None, "sub_confidence": 0.0, "top2": [],
        "is_travel": None,
        "route": None, "route_reason": "", "result": None, "reply": None, "reply_checks": {},
        "timings": None,                       # None -> merge_timings starts a fresh dict
        "history": [{"role": "user", "text": text}],
    }


@dataclass
class Context:
    """Fixed settings for a run (read-only for the nodes, not saved in the state)."""
    today: date = field(default_factory=date.today)
    llm_model: str = "mistral:7b-instruct"
