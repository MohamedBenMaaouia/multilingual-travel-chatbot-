"""The chatbot's entry point: the only thing the UI needs.

    from src.pipeline.chatbot import warm_up, stream_query, handle_query
    warm_up()                                             # once, when the app starts (loads every model)
    for kind, value in stream_query("nheb hotel f jerba", thread_id="user-1"):
        ...                                               # ("token", piece) / ("replace", text) / ("done", (reply, debug))
    reply, debug = handle_query("ghodwa", thread_id="user-1")   # same, without streaming

It wraps the graph (src/pipeline/graph.py): builds it ONCE, starts a turn with new_turn(), passes today's date
(Context), forwards the streamed words, and collects the debug info for the UI's side panel.

Memory: the graph's checkpointer keeps ONE SEPARATE memory per conversation (thread_id). A new thread_id starts
empty; conversations never see each other. The graph is built once because the checkpointer lives inside it:
rebuilding it would empty every conversation's memory. InMemorySaver = RAM only: restarting the app forgets all.
"""
import time
from datetime import date
from functools import lru_cache

from src.pipeline.graph import build_graph
from src.pipeline.state import Context, new_turn


@lru_cache(maxsize=1)
def get_app():
    return build_graph()                       # built once: its checkpointer holds one memory per thread_id


def warm_up():
    """Send one message through the whole graph so every model is loaded (~15 s once) before real users arrive.
    It uses its own throwaway conversation, so no real conversation is affected."""
    t = time.perf_counter()
    handle_query("salam", thread_id="__warm_up__")
    return time.perf_counter() - t


def debug_info(s, total_seconds, first_word_seconds):
    """What the UI's side panel shows: what each stage understood and decided, and how long it took."""
    result = s.get("result") or {}
    return {
        "language": f"{s.get('lang')} ({s.get('lang_confidence', 0):.0%})",
        "conversation_language": s.get("conv_lang"),
        "travel_filter": {True: "travel", False: "other", None: "unknown (filter failed)"}[s.get("is_travel")],
        "subcategory": f"{s.get('subcategory')} ({s.get('sub_confidence', 0):.0%})",
        "places": [f"{p['text']} -> {p['canonical']} ({p['kind']}, {p['role']}, {p['status']})" for p in s.get("places", [])],
        "dates": [f"{d['text']} -> {d.get('date')}" for d in s.get("dates", [])],
        "route": s.get("route"),
        "route_reason": s.get("route_reason"),
        "memory": {"topic": s.get("topic"), "slots": s.get("slots"), "pending": s.get("pending")},
        "tool": {"name": result.get("tool"), "arguments": result.get("arguments"),
                 "defaults_used": result.get("defaults_used"), "data": result.get("data")} if result else None,
        "reply_checks": s.get("reply_checks") or None,
        "timings_s": {**{k: round(v, 3) for k, v in (s.get("timings") or {}).items()},
                      "first_words": round(first_word_seconds, 2) if first_word_seconds else None,
                      "total": round(total_seconds, 2)},
    }


def stream_query(text, thread_id, today=None):
    """Run one user message. Yields ("token", piece) while Mistral writes, ("replace", text) if the reply is
    replaced by a fallback, and finally ("done", (reply, debug))."""
    app = get_app()
    config = {"configurable": {"thread_id": thread_id}}
    context = Context(today=today or date.today())
    t0, first, state = time.perf_counter(), None, None
    for mode, chunk in app.stream(new_turn(text), config, context=context, stream_mode=["custom", "values"]):
        if mode == "custom" and "token" in chunk:
            first = first or time.perf_counter() - t0
            yield "token", chunk["token"]
        elif mode == "custom" and "replace" in chunk:
            yield "replace", chunk["replace"]
        elif mode == "values":
            state = chunk                      # the state after each step; the last one is the final state
    yield "done", (state.get("reply") or "", debug_info(state, time.perf_counter() - t0, first))


def handle_query(text, thread_id, today=None):
    """Same as stream_query, without streaming: returns (reply, debug)."""
    for kind, value in stream_query(text, thread_id, today):
        if kind == "done":
            return value


if __name__ == "__main__":
    # Quick check:  python -m src.pipeline.chatbot
    import json
    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    print(f"warm-up: {warm_up():.1f} s")
    for msg in ["nheb hotel f jerba", "nhar el jem3a, 3 lyeli"]:
        print(f"\nUSER: {msg}\nBOT:  ", end="", flush=True)
        for kind, value in stream_query(msg, thread_id="demo"):
            if kind == "token":
                print(value, end="", flush=True)
            elif kind == "replace":
                print(f"\n[replaced]\n{value}", end="")
            else:
                reply, debug = value
                if debug["route"] == "clarifier":
                    print(reply, end="")
                print("\nDEBUG:", json.dumps({k: v for k, v in debug.items() if k != "tool"}, ensure_ascii=False, indent=1))
