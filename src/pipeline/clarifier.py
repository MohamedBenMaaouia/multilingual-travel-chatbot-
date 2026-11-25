"""Stage 3 - the Clarifier: ask ONE short question for the missing (or doubtful) information.

The Supervisor already decided WHAT is missing and wrote it in `pending` (e.g. {"slot": "date"}).
The Clarifier only turns it into a fixed question in the conversation's language (no model, no LLM: instant,
and the Tunisian questions were approved by the user).

No LangGraph `interrupt` here (deliberate): the Clarifier replies and the turn ends. The user's answer comes back
as a NORMAL message, goes through the Understanding agent ("ghodwa" -> 2026-11-25), and the Supervisor's
rule A puts it in the waiting slot. With `interrupt`, the answer would skip the Understanding agent.
"""
from src.pipeline.state import ChatState

QUESTIONS = {
    "destination": {"en": "Where would you like to go?",
                    "fr": "Où souhaitez-vous aller ?",
                    "tn_latin": "Win t7eb temchi?"},
    "city": {"en": "Which city?",
             "fr": "Quelle ville ?",
             "tn_latin": "Ama mdina?"},
    "place": {"en": "Which country or city?",
              "fr": "Pour quel pays ou quelle ville ?",
              "tn_latin": "Ama blad wala ama mdina?"},
    "date": {"en": "For which date?",
             "fr": "Pour quelle date ?",
             "tn_latin": "Lel ama nhar?"},
    "airline": {"en": "Which airline are you flying with?",
                "fr": "Avec quelle compagnie voyagez-vous ?",
                "tn_latin": "M3a ama compagnie bech tsafer?"},
    "from_currency": {"en": "Which currency do you want to change, and how much?",
                      "fr": "Quelle devise voulez-vous changer, et combien ?",
                      "tn_latin": "Ama 3omla t7eb tbaddel, w 9adech?"},
    "doubtful": {"en": "I didn't understand which place you mean. Could you write its name again?",
                 "fr": "Je n'ai pas compris de quel lieu il s'agit. Pouvez-vous réécrire son nom ?",
                 "tn_latin": "Ma fhemtch ama blasa t9assed. Tnajjem t3awed tekteb esmha?"},
}
QUESTIONS["origin"] = {"en": "Where are you flying from?",          # never REQUIRED today; kept for doubtful origins
                       "fr": "D'où partez-vous ?",
                       "tn_latin": "Men win bech tsafer?"}


def clarifier_node(state: ChatState):
    pending = state["pending"]
    key = "doubtful" if pending.get("doubtful") else pending["slot"]
    question = QUESTIONS[key][state.get("conv_lang") or "en"]
    return {"reply": question, "history": [{"role": "bot", "text": question}]}
