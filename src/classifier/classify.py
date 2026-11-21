"""Module 3 - Subcategory classifier (MLP on TF-IDF), as used by the chatbot.

    from src.classifier.classify import classify
    classify("9adech el billet l <LOC> <DATE> ?")
    -> {"subcategory": "flight_price", "category": "flights", "confidence": 0.93,
        "top2": [("flight_price", 0.93), ("flight_search", 0.05)]}

Input: the MASKED message (places and dates replaced by <LOC> / <DATE>, done by the NER).
The model is trained in notebooks/03_classifier_mlp_tfidf.ipynb and saved to models/classifier/mlp_tfidf.joblib
(models/ is not stored in git: run the notebook to recreate it).
"top2" lets the orchestrator ask the user to clarify when the two best subcategories are close
(e.g. a message asking for a flight AND a hotel price).
"""

from functools import lru_cache
from pathlib import Path

import joblib

from src.labels import SUBCAT_TO_CAT
from src.preprocessing.normalize import normalize

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "classifier" / "mlp_tfidf.joblib"


@lru_cache(maxsize=1)          # load once, then reuse
def _load():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"{MODEL_PATH} not found. Run notebooks/03_classifier_mlp_tfidf.ipynb to train it.")
    return joblib.load(MODEL_PATH)


def classify(text_masked: str) -> dict:
    bundle = _load()
    X = bundle["vectorizer"].transform([normalize(text_masked)])     # SAME cleaning as training
    probs = bundle["mlp"].predict_proba(X)[0]
    ranked = sorted(zip(bundle["mlp"].classes_, probs), key=lambda t: -t[1])
    sub, conf = ranked[0]
    return {"subcategory": sub, "category": SUBCAT_TO_CAT[sub], "confidence": float(conf),
            "top2": [(s, float(p)) for s, p in ranked[:2]]}
