"""Module 2 - Travel filter using the local LLM (decided by the user instead of a One-Class SVM).

    from src.travel_filter.llm_filter import classify_travel
    classify_travel("9adech billet l paris ?")   ->  {"is_travel": True, "raw": "TRAVEL", "seconds": 0.4}

The LLM answers one word, TRAVEL or OTHER. If its answer is unclear we treat the message as travel:
refusing a real customer (false rejection) is the worst mistake for the chatbot.

Two prompt versions (both kept so the report can compare them):
  v1 - a short instruction only (zero-shot). On val: accuracy 88.7%, 7.4% of travel questions refused.
  v2 - a stricter definition + 16 solved examples taken from the TRAIN set (few-shot), and the answer is
       FORCED to be {"label": "TRAVEL"} or {"label": "OTHER"} with a JSON schema (no unclear answers).
  v3 - v2 + one sentence: greetings, thanks and acknowledgements are OTHER. Used by the chatbot
       (src/pipeline/understanding.py). Small-talk check: 14/15 OTHER (v2: 9/15), short travel questions 8/8.
       Not re-measured on val (user's choice).
"""

import json

from src.llm.ollama_client import chat

DEFAULT_MODEL = "mistral:7b-instruct"   # the project's single LLM for every LLM task

PROMPT_V1 = (
    "You are a classifier for a travel agency chatbot. Decide whether the user's message is about travel: "
    "flights, hotels and accommodation, local transport (taxis, car rental, trains, buses, louages), "
    "weather at a destination, things to do at a destination, visas and travel documents, currency exchange, "
    "trip budget, or how to pay for a trip. "
    "Messages may be written in English, French, or Tunisian Arabic in Latin letters (Arabizi, for example "
    "'9adech el billet l paris?' or 'nheb hotel f jerba'). "
    "Answer with exactly one word: TRAVEL or OTHER."
)

PROMPT_V2 = (
    "You are a strict classifier for a travel agency chatbot. Label the user's message TRAVEL or OTHER.\n"
    "TRAVEL = anything a travel agency would answer: flights and airlines, luggage and baggage rules, hotels and "
    "their services, local transport (taxis, car rental, trains, buses, louages), the weather or temperature in a "
    "city, things to do at a destination, visas, passports and vaccines for a trip, currency exchange and "
    "converting amounts of money, trip costs and budgets, and how to pay (cards, cash, bank transfer).\n"
    "OTHER = everything else: cooking, health, school, work, technology, sport, pets, home, music, films, writing "
    "messages, general knowledge, greetings, even when the message mentions a city.\n"
    "Messages are in English, French, or Tunisian Arabic written in Latin letters (Arabizi).\n"
    "Reply with one word only: TRAVEL or OTHER. Never write anything else."
)

# v3 = v2 + one sentence: thanks and acknowledgements ("merci", "ok", "behi") were labelled TRAVEL by v2.
PROMPT_V3 = PROMPT_V2.replace(
    "Messages are in English",
    "Greetings, thanks, goodbyes and short acknowledgements ('ok', 'merci', 'perfect') are OTHER: "
    "they are not a travel request.\nMessages are in English",
)

# Solved examples, all taken from data/processed/train.jsonl (never from val or test).
FEW_SHOT_V2 = [
    ("What's the value of 100 Swiss francs in Tunisian dinars?", "TRAVEL"),
    ("My soup is too salty. How can I fix it?", "OTHER"),
    ("fama orages fi gabes fel weekend?", "TRAVEL"),
    ("fama psychologue behi fi sousse?", "OTHER"),
    ("Quel temps fera-t-il à Tunis demain ?", "TRAVEL"),
    ("Comment transférer mes photos vers un nouvel ordinateur ?", "OTHER"),
    ("Can I put perfume bottles in my checked suitcase?", "TRAVEL"),
    ("nheb recette houmous bel tahini, chnowa les quantités?", "OTHER"),
    ("najjem ne5ou bagage cabine sghir w valise en soute?", "TRAVEL"),
    ("Hey, are you there?", "OTHER"),
    ("Puis-je payer mon hôtel à Tunis avec une carte bancaire étrangère ?", "TRAVEL"),
    ("fama réparateur télé behi fi bizerte?", "OTHER"),
    ("El paiement cash ma9boul?", "TRAVEL"),
    ("kifech n9allel stress 9bal examen?", "OTHER"),
    ("Is there a children's swimming pool at the resort in Zarzis?", "TRAVEL"),
    ("Fama commission ki nbaddel flous?", "TRAVEL"),
]

SYSTEM_PROMPT = PROMPT_V2   # the prompt in use (shown in the notebook)

# v2 output format: Ollama only lets the model write JSON matching this schema
LABEL_SCHEMA = {"type": "object", "properties": {"label": {"type": "string", "enum": ["TRAVEL", "OTHER"]}},
                "required": ["label"]}


def build_messages(text, version=2):
    if version == 1:
        return [{"role": "system", "content": PROMPT_V1}, {"role": "user", "content": text}]
    messages = [{"role": "system", "content": PROMPT_V3 if version == 3 else PROMPT_V2}]   # v2 and v3: same examples
    for example, label in FEW_SHOT_V2:
        messages += [{"role": "user", "content": example},
                     {"role": "assistant", "content": json.dumps({"label": label})}]
    return messages + [{"role": "user", "content": text}]


def classify_travel(text, model=DEFAULT_MODEL, version=2):
    """Return {"is_travel": bool, "raw": the model's answer, "seconds": time taken}."""
    if version == 1:
        raw, seconds = chat(model, build_messages(text, 1), temperature=0.0, max_tokens=5)
        answer = raw.strip().upper()
    else:
        raw, seconds = chat(model, build_messages(text, version), temperature=0.0, max_tokens=20, format=LABEL_SCHEMA)
        try:
            answer = json.loads(raw)["label"]
        except (json.JSONDecodeError, KeyError, TypeError):
            answer = ""                                        # should not happen with the schema
    is_travel = not answer.startswith("OTHER")                 # unclear answer -> travel (safe side)
    return {"is_travel": is_travel, "raw": raw, "seconds": seconds}
