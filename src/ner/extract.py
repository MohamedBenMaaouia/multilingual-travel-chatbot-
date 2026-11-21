"""Module 4 - NER (places and dates) with the fine-tuned XLM-RoBERTa model, as used by the chatbot.

    from src.ner.extract import extract_entities
    extract_entities("nheb billet men tunis l Paris ghodwa")
    -> [{"text": "tunis", "label": "LOC", "start": 16, "end": 21},
        {"text": "Paris", "label": "LOC", "start": 24, "end": 29},
        {"text": "ghodwa", "label": "DATE", "start": 30, "end": 36}]

The model is trained on Google Colab with notebooks/04_ner_xlmr_colab.ipynb and unzipped into models/ner_xlmr/
(not stored in git). It works on the ORIGINAL text (not lowercased: XLM-R is case-sensitive).
The word splitting and the tag-to-span logic are the same as in the notebook.
"""

import re
from functools import lru_cache
from pathlib import Path

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models" / "ner_xlmr"
WORD_RE = re.compile(r"\w+|[^\w\s]")      # words, and every punctuation mark on its own (same as training)


@lru_cache(maxsize=1)          # load once (a few seconds), then reuse
def _load():
    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"{MODEL_DIR} not found. Run notebooks/04_ner_xlmr_colab.ipynb on Colab and unzip "
                                "ner_xlmr.zip into models/ner_xlmr/.")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, dtype=torch.float32)  # saved in fp16
    model.eval()
    return tokenizer, model


def extract_entities(text: str) -> list[dict]:
    tokenizer, model = _load()
    words = [(m.group(), m.start(), m.end()) for m in WORD_RE.finditer(text)]
    if not words:
        return []
    enc = tokenizer([w for w, _, _ in words], is_split_into_words=True, truncation=True, max_length=96,
                    return_tensors="pt")
    with torch.no_grad():
        pred = model(**enc).logits.argmax(-1)[0].tolist()

    # one tag per word = the tag of its first sub-token
    tags, previous = ["O"] * len(words), None
    for pos, wid in enumerate(enc.word_ids()):
        if wid is not None and wid != previous:
            tags[wid] = model.config.id2label[pred[pos]]
        previous = wid

    # BIO tags -> spans (an I- tag that does not continue the previous entity starts a new one)
    spans = []
    for i, tag in enumerate(tags):
        if tag == "O":
            continue
        label = tag[2:]
        continues = tag.startswith("I-") and spans and spans[-1]["label"] == label and spans[-1]["last"] == i - 1
        if continues:
            spans[-1]["end"], spans[-1]["last"] = words[i][2], i
        else:
            spans.append({"label": label, "start": words[i][1], "end": words[i][2], "last": i})
    return [{"text": text[s["start"]:s["end"]], "label": s["label"], "start": s["start"], "end": s["end"]}
            for s in spans]


if __name__ == "__main__":                 # quick manual check:  python -m src.ner.extract
    import time
    for msg in ["nheb billet men tunis l Paris ghodwa", "Quel temps fait-il à La Marsa le 20 décembre ?",
                "flight from Tunis to Kerkennah next Friday", "9adech el euro bel dinar ?",
                "fama louage men kasserine l soussa ba3d ghodwa?"]:
        start = time.perf_counter()
        ents = extract_entities(msg)
        print(f"{(time.perf_counter() - start) * 1000:6.0f} ms | {msg}\n          -> {[(e['text'], e['label']) for e in ents]}")
