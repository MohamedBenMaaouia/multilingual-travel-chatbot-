# Project Plan — Multilingual Travel Agency Chatbot

> **Original title:** Agent conversationnel pour agence de voyage — Chatbot multilingue (français, anglais, arabe, dialecte tunisien)
>
> This file is the full project plan. It is meant to be read by the coding agent (Claude Code) to create `CLAUDE.md` and then build the project step by step together with me.
>
> **Scope decision:** the project covers **3 languages only: English (`en`), French (`fr`) and Tunisian dialect in Latin script / Arabizi (`tn_latin`)**. Modern Standard Arabic and Tunisian in Arabic script are **out of scope**, so everything is in Latin script.

---

## 1. Context and ground rules

### What the project is
A **text → text** chatbot for a travel agency. The user types a travel question in one of these 3 languages:
- English
- French
- Tunisian dialect in **Latin script / Arabizi** (e.g. "nheb nemchi l Djerba", where 3 = ع, 7 = ح, 9 = ق, 5 = خ, 8 = غ)

The chatbot answers **in the same language as the question**.

It is an **academic university project** (AI engineering degree), with no commercial deployment. Out of scope: online payment, firm booking and voice.

### Ground rules (important, follow these over anything in the original brief)
1. **No timelines.** Ignore any duration or planning constraints. I work at my own pace.
2. **No formal methodology.** The original brief mentions CRISP-DM. Ignore it; we don't follow a strict process.
3. **No real external data APIs.** The original brief mentions OpenWeatherMap, ExchangeRate-API, flights and hotels APIs. We do **not** use them. We **generate our own data** and serve it through a **mock data service** that has the same shape as a real API (so it could be swapped later).
4. **Local LLM only:** the LLM runs locally through **Ollama** (Qwen family). No paid or hosted LLM APIs.
5. **Fine-tuning happens on Google Colab** (free GPU). Everything else (light training, inference, the app) runs locally.

### My setup and profile
- 2nd-year AI engineering student, comfortable with Python and ML basics. Explain what you do and why, without heavy jargon, because I want to understand every part, not just get code.
- **Native Tunisian speaker:** I will personally review all Tunisian dialect data (generated text is often "fake Tunisian", i.e. MSA or French with a few dialect words).
- **Local GPU: 4 GB VRAM** (too weak for fine-tuning transformers, and too small to fully fit a 7B model).
- **Ollama installed.** Currently pulled: `qwen2.5:14b` and `mistral:7b-instruct`. Smaller Qwen models (7B, ~3B) still need to be pulled.
  - 14B: too slow for this hardware, so don't use it in the app. At most use it as a quality comparison.
  - 7B (4-bit): ~4.5–5 GB, so it will partly spill to CPU RAM and be slower. Test it.
  - A smaller Qwen (~3–4B) fits fully in 4 GB VRAM and will be much faster.
  - **Decision to make by measurement:** compare a ~3–4B model vs the 7B on speed and answer quality (especially Tunisian), then pick one. The target is a full response in < 2 s, so the LLM choice matters most here.

---

## 2. System architecture

```
User text
   │
   ▼
[1] Language / dialect detection  ──► en | fr | tn_latin
   │
   ▼
[2] Travel filter (One-Class SVM) ──► not travel → generic polite reply, STOP
   │
   ▼
[7] (optional) Complex query segmentation → several sub-queries, each goes through the rest
   │
   ▼
[8] (optional) Tunisian → French translation before analysis
   │
   ▼
[4] NER: places + dates  ──► {LOC: "Djerba", DATE: "2026-12-20"}
   │
   ▼
[3] Category / subcategory classification ──► Destination info / Weather
   │
   ▼
Slot check: required info missing? → clarification question, STOP
   │
   ▼
Mock data service (weather, exchange, flights, hotels) ──► structured data
   │
   ▼
[5] LLM (Ollama, local Qwen) writes the answer in the user's language
   │
   ▼
[6] Chat interface + 👍 / 👎 feedback stored
   │
   └── [9] (optional) sentiment analysis to adapt tone
```

### Walkthrough example
Input: `nheb nemchi l Djerba 20 décembre, chnowa el ta9s?` ("I want to go to Djerba on Dec 20, what's the weather?")
1. Language → `tn_latin`
2. Travel filter → travel ✅
3. NER → LOC = Djerba, DATE = 20 December → normalized `2026-12-20`
4. Category → `destination_info` / `weather`
5. Slots OK → `get_weather("Djerba", "2026-12-20")` → `{temp: 18, condition: "sunny"}`
6. LLM → short friendly answer in Tunisian (Latin script)
7. UI shows the answer and the user clicks 👍/👎

Branches:
- **Non-travel** ("how do I cook couscous?") → generic reply: "I can only help with travel questions…" (in the user's language).
- **Ambiguous / incomplete** ("I want a flight") → clarification question: "Where would you like to go, and when?"

---

## 3. Step 0 — Dataset (the foundation)

### Requirements from the brief
- **≥ 1,000 annotated queries** across the 3 languages
- Class balance checked
- **Double annotation of a sample** (I re-check ~100 rows and we report the agreement rate)
- Cleaning, normalization, **entity masking** (e.g. `température à <LOC> en <DATE>`), embeddings, train/val/test split

### Targets (decided: ~1,500 queries per language, ~4,500 in total)
- ~**1,300 travel queries per language** (~87 per language × subcategory cell, ~3,900 in total)
- ~**200 non-travel queries per language** (~600 in total), for training and evaluating the travel filter
- Reason: with the 15% test split this gives ~13 test examples per language × subcategory cell and ~90 non-travel test queries, enough to trust per-cell and per-language metrics. The original ~1,500 total would have left only ~5 per cell.
- To avoid near-duplicate data, these come from **more templates (~18 per cell)** plus **LLM-generated variations**, not from filling the same few templates many times.
- A set of **multi-intent queries** if we do Module 7

### Language labels (3 classes)
`en`, `fr`, `tn_latin`

### Taxonomy: 5 categories × 3 subcategories (the minimum is 5 / 15; it can grow later)
| Category | Subcategories |
|---|---|
| `flights` | `flight_search`, `flight_price`, `baggage` |
| `accommodation` | `hotel_search`, `hotel_price`, `hotel_facilities` |
| `local_transport` | `car_rental`, `transfer_taxi`, `public_transport` (louage, train, bus) |
| `destination_info` | `weather`, `activities`, `visa_documents` |
| `budget_money` | `exchange_rate`, `trip_cost`, `payment_methods` (info only, no payment) |

### Data format (JSONL, one query per line)
```json
{
  "id": "q_000123",
  "text": "nheb billet l Paris ghodwa",
  "lang": "tn_latin",
  "is_travel": true,
  "category": "flights",
  "subcategory": "flight_search",
  "entities": [
    {"start": 13, "end": 18, "label": "LOC", "value": "Paris"},
    {"start": 19, "end": 25, "label": "DATE", "value": "ghodwa"}
  ],
  "text_masked": "nheb billet l <LOC> <DATE>",
  "source": "template | llm | manual",
  "verified": false,
  "multi_intent": false
}
```
For non-travel rows: `is_travel: false`, and `category`/`subcategory` are `null`.

### How we generate it
1. **Templates** per subcategory and per language, with slots `{LOC}`, `{DATE}`, `{ORIGIN}`, `{CURRENCY}`, etc., filled from lists:
   - places: Tunisian cities (Djerba, Tunis, Sousse, Hammamet, Tozeur, Tabarka, Monastir, Mahdia…) plus common foreign destinations (Paris, Istanbul, Rome, Dubai…), each with the spellings used in each language (e.g. "Djerba" / "Jerba", "Rome" / "roma")
   - dates: absolute ("20 décembre", "Dec 20", "20 decembre") and relative ("demain", "tomorrow", "ghodwa", "ba3d ghodwa", "el weekend jey"…)
   - **Big advantage:** since we insert the entities ourselves, we get **exact entity spans for free** (needed for NER).
2. **LLM-assisted generation** (local Qwen via Ollama) for natural variety: "write 20 varied ways a Tunisian would ask about car rental, in Latin script". For these, entity spans are computed by matching known place/date values in the text. Anything that can't be matched gets flagged for manual review.
3. **Manual review by me**, especially for all Tunisian rows. A small review script should show rows one by one and let me accept/edit/reject, setting `verified: true`.
4. **Non-travel queries:** general topics (cooking, sport, school, tech, health…) in all 3 languages.

### Preprocessing
- Lowercase, trim, collapse spaces
- **Keep Arabizi digits** (3, 7, 9, 5, 8) as they are, because they are letters in `tn_latin`
- Build `text_masked` by replacing entity spans with `<LOC>` / `<DATE>`
- **Balance report:** counts per language, category, subcategory and the language × subcategory grid. Generate more for weak cells.
- **Stratified split** 70 / 15 / 15 (train / val / test), with a fixed random seed. The test set is never used until final evaluation.

### Mock "API" data (also generated in this step)
- `weather.csv`: city, month (or date), temp_min, temp_max, condition, humidity
- `exchange_rates.json`: rates between TND, EUR, USD, GBP, etc.
- `flights.csv`: origin, destination, date, airline, departure time, price, baggage allowance
- `hotels.csv`: city, name, stars, price per night, facilities (pool, wifi, breakfast…)
- Values should be **realistic** (warm summers in Djerba, colder Tozeur nights in winter, plausible prices).

---

## 4. Mandatory modules

### Module 1 — Language / dialect detection
- **Job:** classify the query into `en | fr | tn_latin`.
- **Hard case:** `tn_latin` vs `fr`, because Tunisians code-switch heavily with French (e.g. "nheb billet l Paris" mixes both).
- **Models to compare:**
  - **fastText** supervised classifier (character n-grams), trained locally in seconds
  - **XLM-RoBERTa (base)**, fine-tuned on Colab
- **Metric:** accuracy **per language class**, target **≥ 90%** for each. Also a confusion matrix.
- Report inference time and model size for both, which gives a second benchmark table.

### Module 2 — Travel / non-travel filter
- **Job:** decide whether the query is about travel.
- **Method:**
  1. Sentence embeddings from a multilingual sentence model (e.g. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`; it runs fine on CPU). Can compare with `LaBSE`.
  2. **One-Class SVM** (scikit-learn) trained **only on travel queries**.
  3. Tune `nu` (and `gamma`) on a validation set containing both travel and non-travel queries.
- **Metrics:** precision/recall/F1 for the "non-travel" detection, false rejection rate of real travel queries, broken down per language (dialect may be weaker).

### Module 3 — Category / subcategory classification (main benchmark)
- **Job:** predict category and subcategory.
- **Approach:** start with **flat classification over 15 subcategories** (the category is derived from the subcategory). A hierarchical version (category model, then subcategory model) is an optional comparison.
- Input: `text_masked`, so the model learns the intent and not city names.
- **Model A, from scratch (required by the brief):** MLP, CNN or LSTM with random init, trained only on our data. Options:
  - MLP on TF-IDF features (simplest, strong baseline)
  - MLP on sentence embeddings
  - CNN or BiLSTM on tokens/characters (character-level helps with Arabizi spelling variation)
- **Model B:** XLM-RoBERTa fine-tuned on Colab.
- **Metrics:** accuracy, **macro F1**, confusion matrix, **inference time** per query, **memory / model size**. Target **> 85%**.
- **Deliverable:** benchmark table comparing the models.

### Module 4 — Place and date extraction (NER)
- **Job:** extract `LOC` and `DATE` entities.
- **Options (pick one, or compare):**
  - **spaCy** NER trained from our spans (blank multilingual pipeline)
  - **Hugging Face token classification**: XLM-RoBERTa fine-tuned on Colab (BIO tags)
- **Helpers:**
  - **Gazetteer** (list of known places in all scripts) as a fallback when the model misses one
  - **Date normalization:** `dateparser` for EN/FR, plus a **custom Tunisian dictionary** (lyoum, ghodwa, ba3d ghodwa, el jem3a jeya, el weekend, chhar jey…) → actual ISO dates relative to today
  - Place normalization: "Djerba" / "jerba" / "jarba" → canonical `Djerba` (needed to query the mock data)
- **Metrics:** entity-level precision / recall / F1 per label.

### Module 5 — Response generation + mock data service
**Part A: mock data service.**
- Functions with API-like signatures and JSON-like outputs:
  - `get_weather(city, date)`
  - `get_exchange_rate(from_currency, to_currency, amount=None)`
  - `search_flights(origin, destination, date)`
  - `search_hotels(city, checkin, checkout, max_price=None)`
- Read from the generated CSV/JSON files.
- Optionally exposed as a small **FastAPI** service so it behaves like a real REST API. The README states clearly that it can be swapped for real providers (OpenWeatherMap, ExchangeRate-API…).

**Part B: LLM answer generation (Ollama, local Qwen).**
- The LLM **does not decide anything**. It turns structured info into a natural answer.
- The prompt includes: detected language/script, subcategory, entities, mock data result, and an instruction to answer briefly in the same language/script.
- Few-shot examples of real Tunisian phrasing (Latin script) in the prompt, written/validated by me.
- Keep generation short (limit max tokens) for speed.
- **Fallback templates** for when the LLM is slow or fails, plus the fixed replies (non-travel, clarification) in all 3 languages.

**Orchestrator (pipeline glue).**
- One main function, `handle_query(text, conversation_state) -> response + debug_info`.
- Calls the modules in order and handles the branches (non-travel → stop; missing slots → clarification).
- **Required slots per subcategory** (e.g. `flight_search` needs destination, and origin/date are asked if missing; `weather` needs place, and date defaults to today/soon).
- Keeps short **conversation state**, so if the bot asked "where to?" and the user replies "Djerba", it fills the slot instead of restarting.
- Measures and returns **timing per module** for the response time evaluation.

### Module 6 — Chat interface + user feedback
- **Gradio or Streamlit** chat UI (decide when we get there). Optionally a **FastAPI** backend with a `/chat` endpoint, which covers the "executable API" deliverable.
- 👍 / 👎 under each bot answer → saved to **SQLite** (timestamp, query, response, lang, category, rating, response time).
- **Debug side panel** (for the demo and the jury): detected language, travel score, category/subcategory, entities, mock data used, time per module.

---

## 5. Optional modules

### Module 7 — Complex query segmentation
- Example: "vol pour Djerba et voiture sur place" → ["vol pour Djerba", "location de voiture à Djerba"]
- A binary classifier (single vs multi-intent), then **T5-base** (or mT5 for multilingual) fine-tuned to rewrite into sub-queries.
- Needs generated pairs (complex → list of simple queries).
- Simpler alternative to compare against: ask the local LLM to split.

### Module 8 — Tunisian dialect translation (recommended option)
- Translate `tn_latin` queries to French with the local LLM before analysis, then compare pipeline accuracy with and without translation on Tunisian queries.

### Module 9 — Sentiment analysis
- A multilingual sentiment model (XLM-RoBERTa or multilingual DistilBERT, pretrained) detects frustration and adapts the tone of the answer.
- Measure performance on a small labeled sample, including dialect.

---

## 6. Evaluation (success criteria)

| Component | Metrics | Target |
|---|---|---|
| Query classification (Module 3) | Accuracy, macro F1, confusion matrix | > 85% |
| Language detection (Module 1) | Accuracy per language/dialect | ≥ 90% |
| Full system | Response time; handling of ambiguous queries | < 2 s; asks for clarification |
| Answer quality | Human evaluation on 30 queries (1–5 scale) + average 👍 rate | ≥ 3.5 |

Also:
- **Benchmark table(s):** ≥ 2 models compared on accuracy, inference time and memory (Module 3 at minimum, Module 1 as a bonus).
- **Test scenarios:** a written list of conversations covering every language, every category, non-travel queries, ambiguous queries, and multi-turn clarification.
- An `eval/` script that runs all test sets and prints/saves every metric and table.

---

## 7. Deliverables
- Annotated dataset + a description (datasheet: how it was built, counts, balance, agreement rate)
- Benchmark table(s)
- Runnable API / interface
- Test scenarios
- Git repo: code, `README.md`, `requirements.txt`
- PDF report (≤ 15 pages, school template)
- Demo video (2–3 min)
- Presentation (15 min + questions)

---

## 8. Suggested repository structure
```
travel-chatbot/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── data/
│   ├── lexicons/          # places, dates, Tunisian date words, currencies (all scripts)
│   ├── templates/         # query templates per language × subcategory
│   ├── generated/         # raw generated queries (template + LLM)
│   ├── processed/         # cleaned, masked, verified, split (train/val/test .jsonl)
│   └── mock_api/          # weather.csv, flights.csv, hotels.csv, exchange_rates.json
├── notebooks/             # Colab notebooks for fine-tuning (XLM-R classifier, NER, etc.)
├── models/                # saved trained models (gitignored if large)
├── src/
│   ├── data_gen/          # template filling, LLM generation, review tool
│   ├── preprocessing/     # normalization, masking, splitting, balance report
│   ├── lang_detect/       # Module 1
│   ├── travel_filter/     # Module 2
│   ├── classifier/        # Module 3 (from-scratch + transformer)
│   ├── ner/               # Module 4 + date/place normalization
│   ├── mock_api/          # Module 5A (+ optional FastAPI service)
│   ├── llm/               # Module 5B: Ollama client, prompts, fallbacks
│   ├── pipeline/          # orchestrator, slot logic, conversation state
│   ├── options/           # Modules 7, 8, 9
│   └── app/               # Module 6: UI + feedback storage
├── eval/                  # evaluation scripts, benchmark tables, human-eval sheet
└── tests/                 # unit tests + scenario tests
```

---

## 9. Build order
1. **Dataset:** lexicons, templates, generation, my review, preprocessing, split, balance report. Also the mock API data.
2. **Module 1** (fastText first, then XLM-R on Colab) and **Module 2**.
3. **Module 3** (from-scratch model + XLM-R, benchmark table) and **Module 4**.
4. **Module 5:** mock data service, then the Ollama/Qwen prompt work (test ~3–4B vs 7B for speed and quality), then the orchestrator with clarification logic.
5. **Module 6:** interface + feedback + debug panel.
6. **Options:** 8 first (most useful), then 9 and/or 7 if wanted.
7. **Evaluation:** metrics, benchmark tables, test scenarios, 30-query human evaluation, then the report and video.

---

## 10. How I want to work with the agent
- Build **module by module**, and explain in plain language what each piece does and why before or while writing it.
- Keep code simple and readable. No over-engineering, since this is a university project.
- For anything trained on Colab, produce a **self-contained notebook** in `notebooks/` that I can upload, run, and from which I download the trained model into `models/`.
- **Never** call real external data APIs. All travel data comes from `data/mock_api/`.
- **All Tunisian dialect data must go through my review** before it is used for training.
- Fix random seeds and log results so the metrics in the report are reproducible.
- When a decision is genuinely mine (model choice, UI framework, which options to do), ask me rather than deciding silently.
