"""Stage 1 - the Understanding agent: read the message with all our models and fill the "message card".

A small graph (subgraph) of 4 nodes, each a thin wrapper around a function built and measured earlier:

             +--> language       (fastText, Module 1) -------------------------------+
    START ---+--> entities       (XLM-R NER + rules, Module 4) --> classifier (MLP) -+--> END
             +--> travel_filter  (Mistral, Module 2, prompt v3) ---------------------+

LangGraph runs the graph in rounds ("supersteps"): all the nodes that are ready run at the same time,
and the next round starts when all of them are done.
    round 1: language + entities + travel_filter together  (~1.4 s = the slowest, Mistral)
    round 2: classifier                                     (~5 ms; it needs the places/dates to mask them)
Each node writes ONLY its own keys of the state (see src/pipeline/state.py).
"""
import time
from typing import Annotated, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime

from src.classifier.classify import classify
from src.lang_detect.detect import detect_language
from src.ner.extract import extract_entities
from src.ner.rules import extract_slots
from src.pipeline.state import ChatState, Context, merge_timings
from src.travel_filter.llm_filter import classify_travel


class MessageCard(TypedDict, total=False):
    """What the Understanding agent sends back to the main graph: the message card ONLY.
    Without this output schema, a subgraph sends back its WHOLE state, so `history` came back too and its
    reducer appended it a second time (the history doubled every turn)."""
    lang: Optional[str]
    lang_confidence: float
    places: list
    dates: list
    currencies: list
    counts: dict
    airlines: list
    to_confirm: list
    subcategory: Optional[str]
    category: Optional[str]
    sub_confidence: float
    top2: list
    is_travel: Optional[bool]
    timings: Annotated[dict, merge_timings]


def language_node(state: ChatState):
    t = time.perf_counter()
    lang, conf = detect_language(state["text"])
    return {"lang": lang, "lang_confidence": conf,
            "timings": {"language": time.perf_counter() - t}}


def entities_node(state: ChatState, runtime: Runtime[Context]):
    t = time.perf_counter()
    entities = extract_entities(state["text"])                      # WHERE the places and dates are
    slots = extract_slots(state["text"], today=runtime.context.today, entities=entities)   # WHAT they mean
    return {"places": slots["places"], "dates": slots["dates"], "currencies": slots["currencies"],
            "counts": slots["counts"], "airlines": slots["airlines"], "to_confirm": slots["to_confirm"],
            "timings": {"entities": time.perf_counter() - t}}


def mask(text, places, dates):
    """Replace places and dates by <LOC> / <DATE>, like the classifier's training data (text_masked)."""
    spans = [(p["start"], p["start"] + len(p["text"]), "<LOC>") for p in places]
    for d in dates:                                        # dates have no offset: find their text
        i = text.lower().find(d["text"].lower())
        if i >= 0:
            spans.append((i, i + len(d["text"]), "<DATE>"))
    for start, end, tag in sorted(spans, reverse=True):    # from the end, so earlier offsets stay valid
        text = text[:start] + tag + text[end:]
    return text


def classifier_node(state: ChatState):
    t = time.perf_counter()
    c = classify(mask(state["text"], state["places"], state["dates"]))
    return {"subcategory": c["subcategory"], "category": c["category"],
            "sub_confidence": c["confidence"], "top2": c["top2"],
            "timings": {"classifier": time.perf_counter() - t}}


def travel_filter_node(state: ChatState, runtime: Runtime[Context]):
    t = time.perf_counter()
    try:
        is_travel = classify_travel(state["text"], model=runtime.context.llm_model, version=3)["is_travel"]
    except Exception:                                      # Ollama down / timeout: don't crash the bot
        is_travel = None                                   # None = unknown -> the Supervisor treats it as travel
    return {"is_travel": is_travel, "timings": {"travel_filter": time.perf_counter() - t}}


def build_understanding():
    g = StateGraph(ChatState, context_schema=Context, output_schema=MessageCard)   # sends back the card only
    g.add_node("language", language_node)
    g.add_node("entities", entities_node)
    g.add_node("classifier", classifier_node)
    g.add_node("travel_filter", travel_filter_node)

    g.add_edge(START, "language")                          # 3 edges from START = 3 branches in parallel
    g.add_edge(START, "entities")
    g.add_edge(START, "travel_filter")
    g.add_edge("entities", "classifier")                   # the classifier waits for the places and dates
    g.add_edge(["language", "classifier", "travel_filter"], END)   # done when all branches are done
    return g.compile()


if __name__ == "__main__":
    # Quick check:  python -m src.pipeline.understanding
    from datetime import date
    from src.pipeline.state import new_turn
    understanding = build_understanding()
    print(understanding.get_graph().draw_mermaid())
    ctx = Context(today=date(2026, 11, 24))
    for msg in ["9adech el billet men tunis l jarba nhar el jem3a?", "what is the visa process to europe?",
                "merci barcha", "w f sousse?"]:
        t = time.perf_counter()
        s = understanding.invoke(new_turn(msg), context=ctx)
        print(f"\n{msg}   ({time.perf_counter() - t:.2f} s in total)")
        print(f"  language:    {s['lang']} ({s['lang_confidence']:.0%})")
        print(f"  travel:      {s['is_travel']}")
        print(f"  subcategory: {s['subcategory']} ({s['sub_confidence']:.0%})   2nd: {s['top2'][1][0]}")
        print(f"  places:      {[(p['canonical'], p['kind'], p['role']) for p in s['places']]}")
        print(f"  dates:       {[d['date'] for d in s['dates']]}")
        print(f"  timings:     { {k: round(v, 3) for k, v in s['timings'].items()} }")
