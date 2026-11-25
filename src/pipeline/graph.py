"""The whole chatbot graph (docs/agentic_architecture.md).

    START -> understanding (subgraph) -> supervisor --route--> clarifier         -> END
                                                           -> chat              -> END
                                                           -> flights / hotels / destination / money /
                                                              local_transport (experts) -> responder -> END

Memory between messages: the checkpointer saves the state under a thread_id (one conversation).
"""
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from src.pipeline.chat_agent import chat_agent_node
from src.pipeline.clarifier import clarifier_node
from src.pipeline.experts import EXPERTS
from src.pipeline.responder import responder_node
from src.pipeline.state import ChatState, Context
from src.pipeline.supervisor import supervisor_node
from src.pipeline.understanding import build_understanding


def build_graph(checkpointer=None):
    g = StateGraph(ChatState, context_schema=Context)
    g.add_node("understanding", build_understanding())      # stage 1 (a subgraph used as one node)
    g.add_node("supervisor", supervisor_node)                # stage 2
    g.add_node("clarifier", clarifier_node)                  # stage 3
    g.add_node("chat", chat_agent_node)
    for name, expert in EXPERTS.items():
        g.add_node(name, expert)

    g.add_edge(START, "understanding")
    g.add_edge("understanding", "supervisor")
    # the Supervisor wrote `route`; this edge just follows it (the decision itself is in supervisor.py)
    g.add_conditional_edges("supervisor", lambda s: s["route"], ["clarifier", "chat", *EXPERTS])
    g.add_node("responder", responder_node)                  # stage 4
    g.add_edge("clarifier", END)
    g.add_edge("chat", END)
    for name in EXPERTS:
        g.add_edge(name, "responder")
    g.add_edge("responder", END)
    return g.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


if __name__ == "__main__":
    print(build_graph().get_graph(xray=1).draw_mermaid())    # xray=1: also draw inside the subgraph
