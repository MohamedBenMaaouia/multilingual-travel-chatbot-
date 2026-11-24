# LangGraph — a complete, pedagogical guide

> Written for this project, against **LangGraph 1.2.24** (`langgraph-checkpoint` 4.3, `langchain-core` 1.8,
> `langchain-ollama` 1.1). Every feature in Parts 2–4 was run and checked on that version before being written here.

**How to read this guide**
- **Part 1** — the mental model: what LangGraph is and why it exists.
- **Part 2** — the core building blocks (state, nodes, edges, compile, run).
- **Part 3** — control flow: branches, loops, jumps, parallelism.
- **Part 4** — the "production" features: memory, human-in-the-loop, streaming, subgraphs, reliability, context, time travel.
- **Part 5** — working with LLMs and tools.
- **Part 6** — the architectures, from the most naive to the most advanced, each with a diagram, code, pros and cons.
- **Part 7** — design rules and common mistakes.

---

# Part 1 — The mental model

## 1.1 What problem does LangGraph solve?

An LLM application is rarely "one prompt in, one answer out". Real applications must:
- **do several steps** (understand → look up data → write an answer),
- **choose between paths** (travel question vs small talk vs complaint),
- **loop** (ask the user for missing information, try again after an error, call a tool several times),
- **remember** (earlier messages, earlier answers, the user's preferences),
- **pause and wait for a human** (ask a question, get approval),
- **survive failures** (an LLM call times out, a server is down),
- **show progress** (stream the answer word by word).

Writing all that with plain `if/else` and global variables quickly becomes a tangle. **LangGraph models the application as a graph**:
- **nodes** = the steps (plain Python functions: a model, an LLM call, a database lookup...),
- **edges** = which step comes next (fixed, or decided at run time),
- **state** = one shared object that flows through the graph and that every node reads and updates.

LangGraph then gives you, for free: the execution engine, parallelism, loops with a safety limit, saving the state after every step (memory, pause/resume, crash recovery, "time travel"), streaming, retries, and a drawing of the graph.

## 1.2 Workflow vs agent — the most important distinction

| | **Workflow** | **Agent** |
|---|---|---|
| Who decides the next step? | **Your code** (fixed edges, or `if/else` on the state) | **An LLM** (it chooses the next node or the next tool) |
| Predictability | high: same input → same path | lower: the LLM may choose differently |
| Flexibility with unexpected inputs | limited to what you planned | high |
| Cost / latency | low | higher (every decision is an LLM call) |
| Testing | easy | harder (needs scenario tests) |

**LangGraph does both, and — most importantly — lets you mix them in the same graph.** The best real systems are *hybrids*: deterministic where things are clear, an LLM decides only where judgement is needed. This hybrid idea is the backbone of Part 6.

## 1.3 The vocabulary in one picture

```
                ┌────────────── STATE (shared object) ──────────────┐
                │ messages, language, intent, slots, answer, ...   │
                └──────────────────────────────────────────────────┘
                        ▲ read          │ partial update
                        │               ▼
   START ──► [node A] ──edge──► [node B] ──conditional edge──► [node C] ──► END
                                          └─────────────────► [node D] ──► END

 checkpointer: saves the state after every step  (memory, pause/resume, time travel)
 store:        long-term memory across conversations (user preferences...)
```

---

# Part 2 — The core building blocks

## 2.1 State: the shared memory of one run

The state is declared as a `TypedDict` (or a Pydantic model / dataclass). Each key is a "channel".

```python
from typing import TypedDict

class State(TypedDict):
    text: str          # the user's message
    lang: str          # detected language
    answer: str        # the final answer
```

**Rule:** a node never modifies the state in place. It **returns a dict with only the keys it changes**; LangGraph merges that "partial update" into the state.

## 2.2 Reducers: *how* an update is merged

By default, a returned value **replaces** the old one. A **reducer** changes that rule for one key. You attach it with `Annotated[type, reducer]`:

```python
import operator
from typing import Annotated, TypedDict

class State(TypedDict):
    text: str                                   # default: overwrite
    log: Annotated[list[str], operator.add]     # reducer: new list is APPENDED to the old one
```

If node A returns `{"log": ["a"]}` and node B returns `{"log": ["b"]}`, the final `log` is `["a", "b"]` (tested: `['a', 'bx']`).

Why it matters:
- **Parallel nodes** writing the same key at the same time *need* a reducer (otherwise LangGraph raises an error — it cannot know which value to keep).
- **Conversation history**: `langgraph.graph.add_messages` is a special reducer for chat messages: it appends new messages and *updates* a message if one with the same id is returned (so you can edit or delete messages).
- **Custom reducers**: any function `(old, new) -> merged`, e.g. merging dictionaries of slots, keeping the max of two scores, etc.

```python
def merge_dicts(old: dict, new: dict) -> dict:
    return {**(old or {}), **(new or {})}

class State(TypedDict):
    slots: Annotated[dict, merge_dicts]        # slots filled over several turns accumulate
```

**Ready-made state:** `MessagesState` = a state with one key `messages: Annotated[list, add_messages]`. You can subclass it to add your own keys.

**Input / output schemas:** `StateGraph(State, input_schema=Input, output_schema=Output)` lets the graph accept fewer keys than its internal state and return only the keys you choose (internal keys stay private).

## 2.3 Nodes: the steps

A node is a function that receives the current state and returns a partial update:

```python
def detect_language(state: State) -> dict:
    lang, conf = my_fasttext(state["text"])
    return {"lang": lang}
```

Nodes can be **sync or async** (`async def`). A node can also receive extra arguments: the **runtime** (context, store, stream writer — see 4.6) and the **config** (thread id, etc.).

## 2.4 Edges: what comes next

```python
from langgraph.graph import StateGraph, START, END

builder = StateGraph(State)
builder.add_node("detect_language", detect_language)
builder.add_node("answer", answer)
builder.add_edge(START, "detect_language")       # entry point
builder.add_edge("detect_language", "answer")    # fixed edge
builder.add_edge("answer", END)                  # exit
```

- `START` and `END` are special virtual nodes.
- A node can have **several outgoing fixed edges**: the targets then run **in parallel** (see 3.4).

## 2.5 Compile, then run

```python
app = builder.compile()                  # checks the graph (no dangling nodes...) and builds the engine
result = app.invoke({"text": "9adech el billet l paris?"})   # runs to the end, returns the final state
```

Ways to run a compiled graph:
| method | what it does |
|---|---|
| `invoke(input, config)` | run to completion, return the final state |
| `stream(input, config, stream_mode=...)` | run and yield intermediate results (see 4.3) |
| `ainvoke`, `astream` | async versions (needed for async nodes and timeouts) |
| `batch([...])` | run several inputs |

## 2.6 How execution really works: "super-steps"

LangGraph runs in **super-steps** (an idea from Google's Pregel system):
1. All nodes that are "triggered" run **in the same step, in parallel**.
2. Their updates are applied to the state (through the reducers).
3. The edges decide which nodes are triggered next.
4. Repeat until no node is triggered (or `END`).

After **every super-step**, if a checkpointer is attached, the state is **saved**. That single design choice is what makes memory, pause/resume, crash recovery and time travel possible.

There is a safety limit on the number of steps: `recursion_limit` (default 25 in the config; raise it with `app.invoke(x, {"recursion_limit": 50})`), so an agent stuck in a loop stops with an error instead of running forever.

---

# Part 3 — Control flow

## 3.1 Conditional edges: branching on the state

```python
def route(state: State) -> str:
    return "travel" if state["is_travel"] else "refuse"

builder.add_conditional_edges("filter", route, {"travel": "ner", "refuse": "polite_refusal"})
```

- The routing function reads the state and returns a **key**; the mapping turns the key into a node name.
- The routing function can be **pure code** (a workflow) or can **call an LLM** (an agent) — same API.
- It can return a **list** of node names: they all run in parallel.

Tested: input "hello x" routed to node `bx` → `log = ['a', 'bx']`.

## 3.2 `Command`: update the state AND jump, from inside a node

Instead of a separate routing function, a node can return a `Command`:

```python
from langgraph.types import Command

def supervisor(state: State) -> Command:
    nxt = "flights_agent" if state["intent"].startswith("flight") else "hotels_agent"
    return Command(update={"log": ["supervisor"]}, goto=nxt)

builder.add_node("supervisor", supervisor, destinations=("flights_agent", "hotels_agent"))
```

- `update` = the partial state update; `goto` = the next node(s).
- `destinations=` tells LangGraph which nodes the `Command` may jump to (needed to draw the graph correctly).
- `Command(graph=Command.PARENT, goto=...)` lets a node **inside a subgraph jump to a node of the parent graph** — this is how multi-agent "handoffs" work (Part 6).
- `Command(resume=...)` is also used to resume after an `interrupt` (4.2).

**When to use which:** conditional edges when routing is a separate decision; `Command` when the node that does the work also knows where to go next (typical for supervisors and agent handoffs).

Tested: `router` node with `Command(goto=...)` → `['router', 'by']`.

## 3.3 Loops (cycles)

An edge can point **backwards**: that's a loop. It is how agents work (think → act → observe → think...), and how "ask again until the information is complete" works.

```python
builder.add_conditional_edges("check_slots", lambda s: "ask" if s["missing"] else "answer",
                              {"ask": "ask_user", "answer": "answer"})
builder.add_edge("ask_user", "check_slots")     # back to the check after the user replied
```

Always make sure a loop has an exit, and rely on `recursion_limit` as the last safety net.

## 3.4 Parallelism: fixed fan-out and `Send` (map-reduce)

**Fixed fan-out:** several edges from one node → those nodes run in the same super-step, in parallel; a node with several incoming edges waits for all of them ("fan-in").

```python
builder.add_edge("start_understanding", "detect_language")
builder.add_edge("start_understanding", "travel_filter")     # runs at the same time
builder.add_edge(["detect_language", "travel_filter"], "decide")   # waits for both
```

**Dynamic fan-out with `Send`:** when you only know *at run time* how many parallel tasks there are (e.g. one per intent in a multi-question message). A conditional edge returns a list of `Send(node, private_input)`:

```python
from langgraph.types import Send

def fan_out(state):
    return [Send("handle_intent", {"intent": i}) for i in state["intents"]]

builder.add_conditional_edges("split", fan_out, ["handle_intent"])
```

Each `Send` starts one copy of `handle_intent` with **its own input**; their results are merged back into the main state through a reducer (e.g. `results: Annotated[list, operator.add]`). This is the **map-reduce** pattern.

Tested: `["flight", "hotel"]` → two parallel runs → `['FLIGHT', 'HOTEL']`.

`defer=True` on `add_node` makes a node wait until all other pending branches have finished — useful for a final "aggregator" when branches have different lengths.

---

# Part 4 — The "production" features

## 4.1 Memory 1 — the checkpointer (short-term memory of a conversation)

```python
from langgraph.checkpoint.memory import InMemorySaver      # also: SqliteSaver, PostgresSaver (separate packages)

app = builder.compile(checkpointer=InMemorySaver())
config = {"configurable": {"thread_id": "conversation-42"}}
app.invoke({"messages": [("user", "nheb nemchi l jerba")]}, config)
app.invoke({"messages": [("user", "w 9adech el hotel?")]}, config)   # remembers the first turn
```

- A **thread** = one conversation, identified by `thread_id`. Each thread has its own saved state.
- After every super-step the state is written as a **checkpoint**. Invoking again on the same thread **starts from the saved state**: that's conversational memory for free.
- `InMemorySaver` is lost when the program stops; `SqliteSaver` (`pip install langgraph-checkpoint-sqlite`) keeps conversations in a file; `PostgresSaver` for servers.
- **Durable execution:** if the program crashes in the middle of a run, it can resume from the last checkpoint instead of starting over.

## 4.2 Human-in-the-loop: `interrupt` and resume

`interrupt(value)` **pauses the graph inside a node** and returns `value` to the caller (e.g. a question). The state is saved. Later, the caller resumes with `Command(resume=answer)`; the `interrupt(...)` call then **returns the user's answer** and the node continues.

```python
from langgraph.types import interrupt, Command

def ask_city(state):
    if not state.get("city"):
        city = interrupt({"question": "Win t7eb tsafer?"})     # pause here
        return {"city": city}                                  # continues here after resume
    return {}

app = builder.compile(checkpointer=InMemorySaver())             # interrupts REQUIRE a checkpointer
cfg = {"configurable": {"thread_id": "u1"}}
out = app.invoke({"city": ""}, cfg)        # out["__interrupt__"] -> [Interrupt(value={'question': ...})]
out = app.invoke(Command(resume="Djerba"), cfg)                # -> continues with city = "Djerba"
```

Tested: paused with `{'question': 'Which city?'}`, resumed with "Djerba" → `weather in Djerba`.

Important details:
- When resumed, **the node restarts from its beginning** and `interrupt` returns the resume value. So code *before* `interrupt` runs twice: keep it side-effect free (no "send email" before an interrupt).
- `interrupt(value, response_schema=...)` can describe the expected answer format.
- Static breakpoints: `compile(interrupt_before=["node"], interrupt_after=[...])` pause before/after given nodes (great for debugging or "approve before executing" steps).

Uses: asking for missing information, confirming an ambiguous value ("Tqassed Tunis?"), approval before an important action, escalation to a human agent.

## 4.3 Streaming

`app.stream(input, config, stream_mode=...)` yields results **while** the graph runs:

| `stream_mode` | yields | typical use |
|---|---|---|
| `"values"` | the full state after each step | debugging |
| `"updates"` | only what each node changed `{node: update}` | a live "what is happening" panel |
| `"messages"` | LLM tokens as they are generated (+ metadata of the node) | show the answer word by word |
| `"custom"` | anything a node writes with the stream writer | progress messages ("searching flights...") |
| `"debug"` | very detailed events | deep debugging |

Several at once: `stream_mode=["updates", "messages"]`.

Tested (`"updates"`): `[['a'], ['bx']]` — one item per node that ran.

**Why streaming matters for us:** the full answer may take a few seconds on a local 7B model, but if the first words appear after ~1–2 s, the chatbot *feels* fast.

## 4.4 Subgraphs: graphs inside graphs

A compiled graph can be used **as a node** of another graph:

```python
flights_graph = build_flights_agent().compile()
main.add_node("flights_agent", flights_graph)
```

- If parent and subgraph share state keys, they communicate through them automatically; otherwise wrap the call in a function that translates the state.
- Each subgraph can have its own internal steps, loops, interrupts.
- This is how you build **modular, multi-agent systems**: each "agent" is its own subgraph, tested on its own.

Tested: a subgraph used as node `sub` → `['bx']`.

## 4.5 Reliability: retries, fallbacks, timeouts, caching

All configured per node in `add_node(...)`:

```python
from langgraph.types import RetryPolicy, CachePolicy

builder.add_node("call_llm", call_llm,
                 retry_policy=RetryPolicy(max_attempts=3, initial_interval=0.5),  # retry on errors
                 error_handler=fallback_template,                                   # if it still fails
                 cache_policy=CachePolicy(ttl=300))                                 # reuse identical results
app = builder.compile(cache=InMemoryCache())       # caching needs a cache backend
```

| feature | what it does | tested result |
|---|---|---|
| `retry_policy` | re-runs the node on exceptions, with growing waits (backoff); can filter which errors to retry | a node failing twice then succeeding → ran 3 times, success |
| `error_handler` | another node that runs **instead** when the node fails for good — a built-in fallback | a "broken LLM" node → `['fallback template answer']` |
| `timeout` | stops a node that runs too long | **async nodes only** (sync Python cannot be safely cancelled): async node → `NodeTimeoutError`; sync node → `ValueError` at run time |
| `cache_policy` | identical inputs return the cached result (needs `compile(cache=...)`) | 2 identical calls → node executed once |

For us: retries + error handler + async timeout on every LLM node = the chatbot **never** leaves the user without an answer (fallback template if Mistral is slow or down).

## 4.6 Runtime context and the store (long-term memory)

**Context** = read-only information for a run that is *not* conversation state (user id, settings, the model to use):

```python
from dataclasses import dataclass
from langgraph.runtime import Runtime

@dataclass
class Ctx:
    user_id: str

def remember(state, runtime: Runtime[Ctx]):
    ns = ("users", runtime.context.user_id)          # a "folder" in the store
    runtime.store.put(ns, "prefs", {"lang": "tn_latin", "home_city": "Tunis"})
    prefs = runtime.store.get(ns, "prefs").value
    ...

builder = StateGraph(State, context_schema=Ctx)
app = builder.compile(store=InMemoryStore(), checkpointer=...)
app.invoke(inputs, config, context=Ctx(user_id="u42"))
```

**Checkpointer vs store:**
| | checkpointer | store |
|---|---|---|
| scope | one conversation (thread) | across all conversations of a user |
| content | the full graph state | whatever you choose (preferences, facts, summaries) |
| example | "the user asked about Djerba 2 messages ago" | "this user always writes in Tunisian and leaves from Tunis" |

The store also supports **search** (optionally semantic search with embeddings) over saved items.

Tested: a node wrote and read back `{'lang': 'tn_latin'}` for user `u42`.

## 4.7 Time travel and state editing

With a checkpointer, every step is saved, so you can:
- `app.get_state(config)` — the current state (and which node runs next),
- `app.get_state_history(config)` — all past checkpoints of a thread (tested: 4 checkpoints for a 2-node run with an interrupt),
- `app.update_state(config, values, as_node=...)` — edit the state (e.g. fix a wrong slot) as if a node had produced it,
- re-run from any past checkpoint (pass its `checkpoint_id` in the config) → "what if" exploration, debugging, replaying a conversation with a fixed component.

## 4.8 The Functional API (an alternative syntax)

Instead of declaring nodes and edges, write normal Python functions:

```python
from langgraph.func import entrypoint, task

@task
def double(x): return x * 2          # a unit of work (checkpointed, can run in parallel)

@entrypoint(checkpointer=InMemorySaver())
def flow(x):
    return double(x).result() + 1    # plain Python control flow
```

Tested: `flow(5)` → `11`. Same engine (checkpoints, interrupts, streaming), but no drawable graph. **We will use the Graph API**, because the graph *is* the design we want to show and reason about.

## 4.9 Visualising the graph

```python
print(app.get_graph().draw_mermaid())          # Mermaid text (paste into a Markdown viewer)
app.get_graph().draw_mermaid_png()             # PNG (needs internet or local renderer)
app.get_graph(xray=True)                       # also shows the inside of subgraphs
```

Perfect for the report: the drawing comes from the real code, so it can never be out of date.

---

# Part 5 — LLMs and tools inside LangGraph

## 5.1 Calling an LLM in a node

Any way of calling an LLM works inside a node (our own `src/llm/ollama_client.chat`, for example). LangChain's chat models add conveniences:

```python
from langchain_ollama import ChatOllama

llm = ChatOllama(model="mistral:7b-instruct", base_url="http://127.0.0.1:11434", temperature=0)
```

## 5.2 Structured output: make the LLM return data, not prose

For routing and decisions, never parse free text — force a schema:

```python
from pydantic import BaseModel
from typing import Literal

class Decision(BaseModel):
    action: Literal["answer", "clarify", "small_talk", "refuse", "handoff"]
    reason: str

router_llm = llm.with_structured_output(Decision)
decision = router_llm.invoke(messages)            # -> Decision(action="clarify", reason="...")
```

(Our Module 2 already does the same thing with Ollama's `format=` JSON schema.)

## 5.3 Tools

A **tool** = a Python function the LLM is allowed to call, with a name, a description and typed arguments:

```python
from langchain_core.tools import tool

@tool
def search_flights(origin: str, destination: str, date: str) -> list[dict]:
    """Search flights between two cities on a date (YYYY-MM-DD)."""
    return mock_api.search_flights(origin, destination, date)

llm_with_tools = llm.bind_tools([search_flights, get_weather])
```

The LLM answers either with text or with **tool calls** (`name` + `args`). LangGraph's prebuilt pieces run them:
- `ToolNode([...])` — a node that executes the tool calls found in the last message and appends the results as `ToolMessage`s;
- `tools_condition` — a routing function: "the LLM asked for tools → go to the tools node, otherwise → END".

`mistral:7b-instruct` in our Ollama lists the **tools** capability, so tool calling is technically available locally (reliability with a 7B model must be measured).

## 5.4 The prebuilt agent

`langgraph.prebuilt.create_react_agent(llm, tools)` builds the classic think → act → observe loop in one line. Good for learning and prototypes; for this project we will write the graph by hand to control every step (and newer LangChain versions move this helper to `langchain.agents.create_agent`, which is not installed here).

---

# Part 6 — Architectures, from naive to very advanced

Each level adds one idea to the previous one. For each: the picture, a code sketch, when to use it, and its costs.

## Level 0 — A single LLM call (no graph)

```
user ──► LLM ──► answer
```
```python
answer = llm.invoke(user_message)
```
- **Use when:** a demo, a toy.
- **Problems:** no data (it invents prices and dates), no memory, no control, no recovery. Every serious system starts by moving away from this.

## Level 1 — Linear pipeline (a chain)

```
START ─► detect_language ─► extract_entities ─► classify ─► fetch_data ─► write_answer ─► END
```
```python
for a, b in [("lang", "ner"), ("ner", "cls"), ("cls", "fetch"), ("fetch", "write")]:
    builder.add_edge(a, b)
```
- **Use when:** every input needs exactly the same steps.
- **Pros:** simple, predictable, easy to test, each step measurable.
- **Cons:** cannot react: a non-travel message still goes through everything; missing info → wrong answer.
- *This is roughly what Modules 1–4 are when called one after the other.*

## Level 2 — Router (deterministic branching)

```
START ─► understand ─┬─(not travel)─► polite_refusal ─► END
                     ├─(flight)────► flights ─────────► write ─► END
                     └─(weather)───► weather ─────────► write
```
```python
builder.add_conditional_edges("understand", route_by_intent,
    {"refuse": "polite_refusal", "flight": "flights", "weather": "weather"})
```
- **Use when:** inputs fall into known categories, each with its own handling.
- **Pros:** still fully predictable; cheap; each branch tested separately.
- **Cons:** only the cases you planned; ambiguous inputs fall into the wrong branch.

## Level 3 — Stateful conversation: memory + slot filling + human-in-the-loop

```
START ─► understand ─► merge_with_memory ─► check_slots ─┬─(missing)─► ask_user [interrupt] ──┐
                                                         │                                     │
                                                         │◄──────────── (user replies) ────────┘
                                                         └─(complete)─► fetch ─► write ─► END
           checkpointer = conversation memory (thread_id)
```
- **New ideas:** a checkpointer (memory across turns), a reducer that **merges slots** across turns, `interrupt` to ask for missing info, a loop back after the answer.
- **Use when:** real conversations — "I want a flight" → "where to?" → "Djerba" → "when?" → "tomorrow".
- **Pros:** natural multi-turn dialogue; nothing is lost between messages.
- **Cons:** you must design the slots and the questions; state management becomes important (what to keep, what to reset when the topic changes).

## Level 4 — Tool-calling agent (ReAct loop)

```
            ┌──────────────────────────────┐
            ▼                              │
START ─► agent (LLM) ──(tool calls)──► tools (ToolNode)
            │
            └──(no tool call)──► END
```
```python
builder.add_node("agent", lambda s: {"messages": [llm_with_tools.invoke(s["messages"])]})
builder.add_node("tools", ToolNode(tools))
builder.add_conditional_edges("agent", tools_condition)        # tools or END
builder.add_edge("tools", "agent")                              # observe the result, think again
```
- **New idea:** the **LLM decides** which tools to call, in which order, how many times. *Reason → Act → Observe* (ReAct).
- **Use when:** open-ended requests needing several lookups combined in ways you cannot enumerate ("cheapest way to spend 4 days in Djerba with flight + hotel under 1500 dt").
- **Pros:** flexible, handles unplanned combinations.
- **Cons:** each loop = one LLM call (slow on a local 7B model); a small model may call the wrong tool or invent arguments; needs a step limit, argument validation and evaluation.

## Level 5 — Plan-and-execute and map-reduce (parallel work)

```
START ─► planner (LLM) ─► [Send] ─┬─► worker(task 1) ─┐
                                  ├─► worker(task 2) ─┼─► aggregator ─► write ─► END
                                  └─► worker(task 3) ─┘
```
- **New ideas:** an LLM (or code) **splits** the request into sub-tasks; `Send` runs one worker per sub-task **in parallel**; a reducer collects the results; an aggregator combines them.
- **Use when:** multi-part requests ("flight to Paris AND a hotel AND the weather"); several independent lookups.
- **Pros:** parallel → faster than doing tasks one by one; each worker simple.
- **Cons:** splitting must be reliable; combining results into one coherent answer needs care.

## Level 6 — Supervisor (multi-agent, central coordinator)

```
                     ┌──────────► flights_agent ───┐
START ─► supervisor ─┼──────────► hotels_agent ────┼──► (back to supervisor) ─► ... ─► END
       (LLM or code) └──────────► weather_agent ───┘
```
```python
def supervisor(state) -> Command:
    decision = router_llm.invoke(state["messages"])     # structured output
    return Command(goto=decision.next_agent, update={...})
# each agent is a subgraph; after working it returns to the supervisor (edge agent -> supervisor)
```
- **New ideas:** **specialist agents** (each a subgraph with its own tools, prompts and slots) + one **supervisor** that decides who works next, possibly several in turn, then decides when to finish.
- **Use when:** many domains; each domain complex enough to deserve its own agent.
- **Pros:** modular (add an agent without touching the others); each agent small and testable; the supervisor's view of the whole conversation handles follow-ups ("and the hotel there?").
- **Cons:** every supervisor decision is an LLM call; a weak supervisor sends work to the wrong agent; more moving parts.
- (The separate `langgraph-supervisor` package offers a prebuilt version; building it by hand teaches more.)

## Level 7 — Hierarchical teams

```
                         ┌─► transport_supervisor ─┬─► flights_agent
START ─► top_supervisor ─┤                         └─► car_taxi_agent
                         └─► destination_supervisor ┬─► weather_agent
                                                    └─► activities_agent
```
- **New idea:** supervisors of supervisors (subgraphs inside subgraphs). Each level sees only what it needs.
- **Use when:** dozens of agents; very large systems.
- **Pros:** scales to big organisations of agents.
- **Cons:** more LLM decisions per request (latency, cost); overkill for 6 domains.

## Level 8 — Swarm / handoffs (decentralised)

```
flights_agent ──(handoff: "the user now asks about hotels")──► hotels_agent ──► ...
```
```python
def flights_agent(state) -> Command:
    if user_switched_topic(state):
        return Command(goto="hotels_agent", graph=Command.PARENT, update={...})   # hand over control
```
- **New idea:** no central boss; the **active agent hands the conversation to another agent** when needed (`Command(graph=Command.PARENT, goto=...)`). The graph remembers which agent is active, so the next user message goes straight to it.
- **Use when:** long conversations that stay in one domain for a while (customer-service style).
- **Pros:** fewer LLM calls than a supervisor (no boss decision every turn); natural "transfer".
- **Cons:** harder to keep a global view; risk of agents bouncing the user between them.
- (The `langgraph-swarm` package offers a prebuilt version.)

## Level 9 — Reflection, guardrails and self-correction (evaluator-optimizer)

```
... ─► write_answer ─► validator ─┬─(ok)───────────────► END
            ▲                     └─(problem: wrong language / invented number / too long)
            └──────────── revise (max 1–2 times) ◄──────┘           └─(still bad)─► fallback_template
```
- **New idea:** a **checker** node looks at the draft answer and sends it back for revision if it fails a test. The checks can be **code** (is every number in the answer present in the tool data? does fastText say the answer is in the user's language? length?) or **another LLM call** (LLM-as-judge).
- **Use when:** the cost of a wrong answer is high (invented prices, answering in the wrong language).
- **Pros:** catches hallucinations and format errors before the user sees them.
- **Cons:** extra latency on revisions (bounded by a max number of attempts).

## Level 10 — Production-grade hybrid ("deterministic core, agentic edges")

This combines everything above into the kind of design used in real assistants:

```
          ┌──────────────── PERCEPTION (parallel, fast ML) ──────────────────┐
START ─►  │ language │ NER + rules │ classifier │ LLM understanding (JSON)    │
          └───────────────────────────────┬────────────────────────────────────┘
                                          ▼
                     DIALOGUE MANAGER (policy: code first, LLM only when unsure)
        ┌───────────┬────────────┬───────┴─────┬──────────────┬──────────────┐
        ▼           ▼            ▼             ▼              ▼              ▼
   specialist   multi-intent   clarify      small talk /   out of scope   human
   agents       (Send, map-    (interrupt)  greetings      (polite)       handoff
  (subgraphs)    reduce)                                                  (interrupt)
        └───────────┴──────────── composer (LLM, streamed) ─► validator ─► END
   + checkpointer (conversation) + store (user profile) + retries / fallbacks / async timeouts
   + per-node timing and debug info + graph drawing + scenario tests + time-travel debugging
```

Key principles:
1. **Fast deterministic path for clear cases** (most messages): no extra LLM decision.
2. **LLM judgement only where it adds value**: ambiguous, multi-intent, follow-ups, off-script messages.
3. **Every LLM node has a fallback**, so the user is never left without an answer.
4. **State is designed, not improvised**: what is per-turn, what persists in the conversation, what persists across conversations.
5. **Observable**: every decision and its timing is visible (debug panel, traces), every path tested by scenarios.

The concrete version of Level 10 for our chatbot is in [`agentic_architecture.md`](agentic_architecture.md).

---

# Part 7 — Design rules and common mistakes

**State design**
- Keep the state **small and typed**; store ids and structured data, not huge blobs.
- Separate **per-turn** keys (reset each message) from **conversation** keys (kept) and **profile** data (store).
- Every key written by parallel nodes needs a **reducer**.

**Control flow**
- Prefer **code routing** when the rule is clear; use **LLM routing with structured output** only for judgement calls.
- Every loop needs an exit condition **and** a maximum count; keep `recursion_limit` as a last resort.
- Give every LLM decision a **default** when the LLM's output is invalid.

**Reliability**
- Retries for transient errors, **error handlers** for fallbacks, **async + timeout** for slow LLM nodes.
- Code before an `interrupt()` re-runs on resume → keep it side-effect free.

**Testing**
- Unit-test each node as a plain function (it is one).
- Scenario tests: scripted multi-turn conversations run against the compiled graph with a fresh `thread_id`; assert on the path taken (`stream_mode="updates"`) and on the final state.
- Use time travel to replay a bad conversation after fixing a node.

**Latency (important on local hardware)**
- Count LLM calls per message: each costs ~0.5–3 s on our GPU. Put independent work **in parallel**, stream the final answer, and cache identical lookups.
