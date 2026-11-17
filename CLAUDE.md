# CLAUDE.md

Guidance for Claude Code when working in this repository.
The full spec is in [.claude/PROJECT_PLAN.md](.claude/PROJECT_PLAN.md). Read it when you need details. This file holds the rules and facts that apply in every session.

## What this project is

A **text-to-text multilingual travel agency chatbot**, built as an academic project for an AI engineering degree. It will never be deployed commercially. The user asks a travel question and the bot answers **in the same language and script** as the question.

**3 languages only** (use these exact label strings everywhere):
`en` (English) · `fr` (French) · `tn_latin` (Tunisian dialect in Latin script / Arabizi: 3=ع, 7=ح, 9=ق, 5=خ, 8=غ)

Out of scope: Modern Standard Arabic, Tunisian in Arabic script (everything is Latin script), payment, real bookings, voice. The original brief and older notes mention 5 language classes; ignore that.

## Hard rules (these override anything in the original brief)

1. **No real external data APIs.** Never call OpenWeatherMap, ExchangeRate-API, flight or hotel APIs. All travel data comes from generated files in `data/mock_api/`, served through mock functions with API-like signatures so they could be swapped for real providers later.
2. **Local LLM only.** Use Ollama with Qwen models. No paid or hosted LLM APIs.
3. **Transformer fine-tuning runs on Google Colab.** Each fine-tuning job gets a **self-contained notebook** in `notebooks/` that the user uploads, runs, and downloads the model from into `models/`. Everything else (fastText, sklearn, the from-scratch models, inference, the app) runs locally.
4. **Tunisian dialect quality is the user's call.** The user is a native speaker. They approved all tn_latin lexicons and templates, and decided **not** to review the generated rows one by one: the tn_latin data is accepted as is. Don't ask about a row-by-row review again, and don't build a review tool. New hand-written Tunisian text (e.g. few-shot examples in LLM prompts) should still be shown to the user. The brief separately asks for a **double-annotation sample (~100 rows) with an agreement rate**; that is a small evaluation task for later, not a full review.
5. **No timelines and no formal methodology** (ignore CRISP-DM and any schedule in the original brief).
6. **Ask, don't decide silently**, when a choice is the user's to make (see "Open decisions" below).

## How to work with the user

- They are a 2nd-year AI engineering student who is comfortable with Python and ML basics. **Explain what each piece does and why, in plain language**, before or while writing it. They want to understand the whole thing, not just receive code.
- Build **module by module**, following the build order below. Don't jump ahead.
- Keep code **simple and readable**. Don't over-engineer: no unnecessary abstractions, frameworks or config layers.
- **Reproducibility:** fix random seeds (use `42` unless there's a reason not to) and log metrics to files so the numbers in the report can be regenerated.

## Hardware and environment

- Windows 11, PowerShell, Python 3.11. Git repo (branch `main`).
- **Local GPU: 4 GB VRAM**, which is too weak for transformer fine-tuning and too small to fully hold a 7B model.
- Only **~28 GB free** on C:. The user does **not** want to download more models for now, so work with what exists:
  - Ollama: `qwen2.5:14b` (9 GB) and `mistral:7b-instruct` (4.4 GB). No Qwen 7B or 3B is pulled.
  - Hugging Face cache: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (the Module 2 embedding model) and `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
- The 14B is too slow for the app (target: a full response in **< 2 s**), but it is fine for **offline** jobs like dataset generation, where speed doesn't matter. A 7B (4-bit, ~4.5 GB) spills partly to CPU. A ~3B would fit in VRAM. The app LLM is chosen **by measurement** (speed + answer quality, especially Tunisian) among the available models. Suggest pulling a smaller Qwen only if the measurements show it's needed, and only with the user's OK.

## Pipeline

```
text → [1] language detect → [2] travel filter (One-Class SVM) ──not travel──► polite generic reply (user's language)
     → [7] (opt) split multi-intent → [8] (opt) Tunisian→French translation
     → [4] NER (LOC, DATE) + normalization → [3] subcategory classifier
     → slot check ──missing──► clarification question
     → mock data service → [5] LLM writes the answer → [6] UI + 👍/👎 feedback   ([9] opt sentiment → tone)
```

- Orchestrator entry point: `handle_query(text, conversation_state) -> (response, debug_info)`. It returns **timing per module** and keeps short conversation state, so a reply like "Djerba" fills the pending slot instead of restarting the conversation.
- **The LLM decides nothing.** It only turns structured data (lang, subcategory, entities, mock result) into a short natural answer. Few-shot Tunisian examples in prompts must come from or be validated by the user. Limit max tokens. Always have **fallback templates** in all 3 languages, for when the LLM is slow or fails and for the fixed replies (non-travel, clarification).

## Taxonomy (5 categories × 3 subcategories)

| Category | Subcategories |
|---|---|
| `flights` | `flight_search`, `flight_price`, `baggage` |
| `accommodation` | `hotel_search`, `hotel_price`, `hotel_facilities` |
| `local_transport` | `car_rental`, `transfer_taxi`, `public_transport` |
| `destination_info` | `weather`, `activities`, `visa_documents` |
| `budget_money` | `exchange_rate`, `trip_cost`, `payment_methods` |

The classifier is **flat over the 15 subcategories**, and the category is derived from the subcategory. It is trained on `text_masked` so the model learns intent rather than city names.

## Data conventions

JSONL, one query per line:

```json
{"id": "q_000123", "text": "nheb billet l Paris ghodwa", "lang": "tn_latin",
 "is_travel": true, "category": "flights", "subcategory": "flight_search",
 "entities": [{"start": 13, "end": 18, "label": "LOC", "value": "Paris"},
              {"start": 19, "end": 25, "label": "DATE", "value": "ghodwa"}],
 "text_masked": "nheb billet l <LOC> <DATE>",
 "source": "template", "verified": false, "multi_intent": false}
```

- `source` ∈ `template | llm | manual`. Non-travel rows have `is_travel: false` and `category`/`subcategory` set to `null`.
- Entity spans are character offsets into `text` (end exclusive). Labels are `LOC` and `DATE` only.
- **Targets (decided by the user): ~1,500 queries per language, ~4,500 in total** = ~1,300 travel per language (~87 per language × subcategory cell) + ~200 non-travel per language. Volume must come from **variety** (≈18 templates per cell + LLM variations), not from filling the same templates over and over.
- **Preprocessing:** lowercase, trim, collapse spaces. **Never strip Arabizi digits (3, 7, 9, 5, 8)**, because they are letters in `tn_latin`.
- **Split (decided): template-aware + stratified 70/15/15** train/val/test, seed 42. All rows from the same `template_id` go to the same set; GPT rows are their own group. Stratify by language × subcategory (non-travel as its own class). **Never touch the test set until final evaluation.** Anything that *learns from data* (TF-IDF vocabulary, scalers, thresholds) is fitted on train only.
- Place names need canonical forms (jerba / jarba / Djerba → `Djerba`) to query mock data. Dates are normalized to ISO, using `dateparser` plus a custom Tunisian relative-date dictionary (lyoum, ghodwa, ba3d ghodwa, el weekend jey, chhar jey…).
- Mock data must be **realistic** (hot Djerba summers, cold Tozeur winter nights, plausible prices).

## Mock data service

Signatures (read from `data/mock_api/`; FastAPI wrapper optional):
- `get_weather(city, date)`
- `get_exchange_rate(from_currency, to_currency, amount=None)`
- `search_flights(origin, destination, date)`
- `search_hotels(city, checkin, checkout, max_price=None)`

## Repository layout

The repo root is this folder (`travel chatbot/`).

```
data/{lexicons,templates,generated,processed,mock_api}/
notebooks/        # Colab fine-tuning notebooks (self-contained)
models/           # trained models (gitignore large files)
src/{data_gen,preprocessing,lang_detect,travel_filter,classifier,ner,mock_api,llm,pipeline,options,app}/
eval/             # metrics scripts, benchmark tables, human-eval sheet
tests/            # unit + scenario tests
```

## Modules and targets

| # | Module | Approach | Target / metric |
|---|---|---|---|
| 1 | Language detection (`en`/`fr`/`tn_latin`; hard case: `tn_latin` vs `fr` code-switching) | fastText (char n-grams) vs XLM-R base (Colab) | ≥ 90% accuracy **per class**, confusion matrix, inference time + size |
| 2 | Travel filter | multilingual MiniLM embeddings (optionally LaBSE) + One-Class SVM trained on travel only, tune `nu`/`gamma` on val | P/R/F1 for non-travel, false rejection rate, per language |
| 3 | Subcategory classifier | from-scratch model (MLP on TF-IDF / embeddings, or char CNN/BiLSTM) vs XLM-R (Colab) | > 85% accuracy, macro F1, inference time, memory → **benchmark table** |
| 4 | NER | spaCy blank multilingual and/or XLM-R token classification (BIO), with a gazetteer fallback | entity-level P/R/F1 per label |
| 5 | Mock service + LLM + orchestrator | see above | full response < 2 s |
| 6 | UI | Gradio or Streamlit chat, 👍/👎 to SQLite, debug side panel (lang, travel score, category, entities, mock data, per-module timing) | ≥ 3.5/5 human eval on 30 queries |
| 7–9 | Options | 8 = Tunisian translation via LLM (do first), 9 = sentiment, 7 = multi-intent splitting (mT5 vs LLM) | — |

## Build order

1. **Dataset**: lexicons → templates → template filler → external generation (GPT) + import → preprocessing → balance report → split. Generate the mock API data too.
2. Module 1 (fastText first, then XLM-R) and Module 2.
3. Module 3 (both models + benchmark table) and Module 4.
4. Module 5: mock service → LLM prompt work (benchmark the available models) → orchestrator with slots and clarification.
5. Module 6: UI, feedback, debug panel.
6. Options: 8, then 9 and/or 7.
7. Evaluation: `eval/` script that runs all test sets, test scenario list, 30-query human eval, then report and demo.

## Open decisions (ask the user, don't pick silently)

- **One model per module (decided by the user):** don't train two models to compare them. When a module starts, the user says which approach to use (e.g. fastText *or* XLM-R for language detection); build only that one. The approaches in the Modules table are options, not a to-do list. Note: the original brief asks for a benchmark table with ≥ 2 models; the user has chosen to override that.
- Which LLM the app uses (decide from measurements; see Hardware)
- Gradio vs Streamlit
- NER: spaCy vs XLM-R, or compare both
- Which optional modules to build (7, 8, 9)

## Current status

**Step 1, Dataset**, in progress. Sub-steps: lexicons → templates → template filler → LLM generation → review tool → non-travel queries → preprocessing/balance/split → mock API data.

- ✅ Lexicons in `data/lexicons/` (`places.json`, `dates.json`, `currencies.json`, `airlines.json`). The user accepted the tn_latin spellings.
- ✅ Shared label constants in `src/labels.py`.
- ✅ Templates in `data/templates/{en,fr,tn_latin}.json`: 273 per language, 18–19 per subcategory. Template rules: no literal date words, and no preposition directly before `{DATE}`. In `tn_latin.json` the first 8–9 per list are user-approved; the newer ones are **waiting for user review**.
- ✅ Template filler `src/data_gen/fill_templates.py` (default 50 per cell) → `data/generated/template_queries.jsonl` (2,158 rows, 0 span errors). Slot-poor cells (baggage, hotel_facilities, payment_methods, visa_documents) reach only 37–49; the LLM step tops every cell up to ~87.
- ✅ All templates (incl. the new tn_latin ones) approved by the user.
- ❌ Local-Qwen generation was **dropped** by the user: too slow (~3.3 tok/s), weak Tunisian, and few-shot examples pull it back toward the template patterns. `llm_generate.py` was deleted.
- ✅ Instead, the user generates the extra queries with an external model (GPT) using `data/external/gpt_prompt.md`. That's 40 per travel cell plus 220 non-travel per language, written to `data/external/gpt_queries.jsonl`, one JSON per line: `{"lang", "subcategory" (or "non_travel"), "text"}`. Mention this in the datasheet: "generated with GPT, reviewed by a native speaker".
- ✅ `src/data_gen/entity_matcher.py`: `Matcher(lang).find_entities(text)` labels LOC/DATE from the lexicons and rejects rows with unlabeled dates or unlisted places (`strict=False` skips that check for non-travel rows).
- ✅ `src/data_gen/import_external.py`: `gpt_queries.jsonl` (2,559 lines) → `data/generated/external_queries.jsonl` (2,361 rows, 0 span errors) + `external_rejected.jsonl` (198 rows: 97 with Arabic letters, 42 with unlabeled dates/places, 58 over the cell cap, 1 duplicate). It caps travel cells at 87 (templates + GPT) and keeps all non-travel rows.
- **Totals: ~1,500 per language** (en 720 template + 552 GPT travel + 227 non-travel; fr 717 + 563 + 234; tn_latin 721 + 559 + 226). 14 slot-poor cells are at 76–86 instead of 87, which is acceptable.
- ✅ The user decided: no row-by-row review of the Tunisian rows; the data is accepted.
- ✅ `src/preprocessing/build_dataset.py`: merges both sources, cleans safely (no lowercasing; each model normalizes its own input), re-checks spans, dedups, assigns ids `q_000001…`, and does the template-aware stratified split. Outputs `data/processed/{dataset,train,val,test}.jsonl` + `balance_report.md`. Result: **4,519 rows → train 3,161 / val 684 / test 674**, 11–14 test rows per language × subcategory cell, 34–35 non-travel test rows per language; 0 templates and 0 sentences shared across splits.
- ✅ `src/mock_api/generate_data.py` → `data/mock_api/`: `weather.csv` (43 cities × 12 months of climate averages: temp min/max, condition sunny/partly_cloudy/rainy/snowy/windy_sandy, rain days, humidity, sea temp), `hotels.csv` (176 fictional hotels: stars, low/high-season price in TND, high-season months, board, facilities, rating), `flights.csv` (140 weekly-timetable rows: flight_no, airline, IATA origin/destination, days_of_week 0=Mon, time, duration, stops/via, season all_year/summer/winter, base price in TND, cabin/checked bag kg, extra-bag price), `exchange_rates.json` (1 unit → TND). Cities map to airports through `places.json`'s `airport` field.
- **Step 1 (Dataset) is complete.**
- **Module 1 (language detection): the user chose fastText.** In progress.

Note for Module 5: there is no mock data for `activities`, `visa_documents`, `payment_methods`, `public_transport`, `car_rental` or `transfer_taxi` (the plan only lists weather, exchange, flights and hotels). Decide with the user whether to add small static info files or let the LLM answer generally with a disclaimer.

Extra fields beyond the plan's schema: each entity has `norm` (canonical city, or a date rule like `+1d`, `weekday:4`, `md:12-20`, `month:10`), used to evaluate Module 4 normalization. Each row has `template_id`, which can be used for a template-aware split so test sentences don't share templates with train (decide with the user at the split step).

## Conventions

- Run scripts as modules from the repo root: `python -m src.data_gen.fill_templates`. Imports are `from src.labels import ...`.
- Put temporary check scripts in the session scratchpad, not in the repo.
