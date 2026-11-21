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

- Windows 11, PowerShell, Python 3.11. Git repo (branch `main`), pushed to GitHub: `https://github.com/MohamedBenMaaouia/multilingual-travel-chatbot-.git` (remote `origin`).
- **Virtual environment (use it for everything):** `C:\Users\medbe\venvs\travel-chatbot`, deliberately **outside OneDrive** so OneDrive doesn't sync thousands of package files. Run scripts with `C:\Users\medbe\venvs\travel-chatbot\Scripts\python.exe -m src....`. Packages are pinned in `requirements.txt`; add new ones there with exact versions. **numpy is pinned < 2** because fastText's `predict()` breaks with numpy 2. The Jupyter kernel is registered as **"Python (travel-chatbot)"**, and `.vscode/settings.json` (gitignored, machine-specific) points VS Code to the venv.
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

- **Orchestrator = LangGraph (decided by the user).** Each module is a **node**; the branches (not travel → polite reply, missing slot → clarification) are **conditional edges** defined by us, so the flow is deterministic. LangGraph's shared state holds the conversation state. The LLM node only phrases the answer; never use agents or LLM-chosen edges. Draw the graph for the report. Add `langgraph` (and `langchain-ollama` if used) to `requirements.txt` with pinned versions at Module 5.
- Orchestrator entry point: `handle_query(text, conversation_state) -> (response, debug_info)`, wrapping the compiled graph. It returns **timing per module** and keeps short conversation state, so a reply like "Djerba" fills the pending slot instead of restarting the conversation.
- **Module 2 travel filter = the local LLM (decided by the user)**, not the One-Class SVM: "travel" is too broad and fuzzy for a small classifier. The LLM is prompted to answer one word (travel / other) with temperature 0 and a tiny `num_predict`. It is evaluated like a model on val (precision/recall/F1 for non-travel, false rejection rate, per language). Few-shot examples must come from train only. The brief names a One-Class SVM; the report should state the override and the reason. Speed matters: each message now needs two LLM calls (filter + answer).
- **The LLM decides nothing else.** Apart from the travel/other decision above, it only turns structured data (lang, subcategory, entities, mock result) into a short natural answer. Few-shot Tunisian examples in prompts must come from or be validated by the user. Limit max tokens. Always have **fallback templates** in all 3 languages, for when the LLM is slow or fails and for the fixed replies (non-travel, clarification).

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
notebooks/        # local training/evaluation notebooks (VS Code) + Colab fine-tuning notebooks (self-contained)
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
- **The LLM (decided by the user): `mistral:7b-instruct`, one model for every LLM task** (travel filter, answer writing, optional translation). `qwen2.5:14b` is not used. Ollama listens on **`http://127.0.0.1:11434`**: use `127.0.0.1`, not `localhost`, which can resolve to IPv6 and fail. The server must be running (`ollama serve` or the Ollama app). The GPU is an RTX 2050 with 4 GB (~2.7 GB free when last checked), so the model runs partly on CPU.
- (Older note, kept for context.) **Idea to propose at Module 5:** fine-tune a small model (e.g. Qwen2.5 1.5B/3B) with **QLoRA on Colab** on a few hundred good example answers (Tunisian ones written or validated by the user), then convert it for Ollama. It would fit in 4 GB VRAM (faster, toward < 2 s) and be better at Tunisian. Costs: example answers to write, more complex steps, a ~2 GB download.
- **Module 3 classifier = MLP on TF-IDF (decided by the user).** Built slowly in `notebooks/03_classifier_mlp_tfidf.ipynb` (it is a training module). Input: `text_masked` of travel rows, cleaned with `normalize()`; output: one of 15 subcategories (category derived). TF-IDF is fitted on train only. Single-intent; multi-intent messages are handled later (ask the user to clarify when the top 2 probabilities are close, or Module 7).
- Gradio vs Streamlit
- **Module 4 NER = XLM-RoBERTa base token classification (decided by the user)**, fine-tuned on **Google Colab** (self-contained notebook `notebooks/04_ner_xlmr_colab.ipynb`; the user uploads, runs, and downloads the model into `models/ner/`). Labels: BIO with LOC/DATE (O, B-LOC, I-LOC, B-DATE, I-DATE); input = original text (not masked); first-subword labelling (others -100); entity-level P/R/F1 with seqeval. Around the model, rules: lexicon + fuzzy matching for places, date normalization rules, rules for currencies/amounts/counts/airlines/origin-vs-destination. **No slang list** (user decision): if a needed amount/currency is not found, the bot asks. Ambiguous currencies follow the country of the mentioned place, else a sensible default. Plan an NER challenge set with places/dates **outside** the lexicons (val/test only contain lexicon entities, so they overestimate).
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
- **Module 1 (language detection): the user chose fastText.** In progress. Installed `fasttext-wheel` and `matplotlib`. Done: `src/preprocessing/normalize.py` (NFC, lowercase, straight apostrophes, spaces around punctuation; keeps digits, accents, in-word ' and -) and `src/lang_detect/prepare_data.py` → `data/processed/fasttext_lang/{train,val}.txt` (train shuffled with seed 42; no test file yet). Notebook `notebooks/01_lang_detect_fasttext.ipynb` (cells appended with scratchpad scripts so the user's outputs are kept) has sections 0–7: setup, data look, language clues, first training (words only), accuracy (val 99.85%), loss + learning curves (figures in `eval/figures/`), character pieces (slow start: right-by-a-whisker underfitting at lr 0.1), and the tuning grid (words/pieces × lr 0.1/0.5/1.0 × epochs 5/10/25 — all 99.4–99.85% on val, so val cannot separate them). Findings: content-free messages like `paris ?` → Tunisian with high confidence (default-class shortcut, poor calibration). Decision: handle in the orchestrator with the conversation's language, plus a deliberate default for a first message with no evidence. **Challenge set** drafted at `data/challenge/lang_challenge.jsonl`: 60 tricky messages (typos, sms style, no accents, code-switching, short), used only for choosing settings, never as the final test. The user approved the tn_latin lines. Section 8 scores all 18 settings on it (55–59/60). **Final settings (chosen by the user): whole words only, `lr=1.0`, `epoch=10`, `dim=50`, `bucket=200000`, `thread=1`, `seed=42`.** Results: challenge 59/60, val 99.71% (682/684), train ≈ 100%. On `paris ?` it gives en 62% / tn 37% (not over-confident), while most other settings say tn 83–98%. The same setting at 25 epochs is practically identical (tied 59/60, val loss 0.0047 vs 0.0053, `paris ?` en 52%); the user chose to keep 10 epochs. Section 9 shows the final model's confusion matrices (val + challenge) and precision/recall/F1 per language. Remaining val mistakes: `flight nabeul zarzis today` (en→tn) and `dossier visa paris quels papiers` (fr→en); challenge mistake: `vol direct tunis istanbul svp` (fr→tn). Section 10: diagnosis = **good fit** (no under/overfitting; mild over-confidence; remaining errors are content-free keyword messages). Section 11 saves the model to `models/lang_detect/fasttext_lang.bin` (gitignored by the user's choice; recreate it by running the notebook) and the metrics to `eval/metrics/lang_detect_fasttext.json`. The user chose **not** to measure speed/size.
- ✅ `src/lang_detect/detect.py`: `detect_language(text) -> (lang | None, confidence)`. It loads the model once (`lru_cache`), uses `normalize()`, caps confidence at 1.0, and returns `(None, 0.0)` for empty text. The low-confidence rule (keep the conversation's language) belongs in the orchestrator.
- **Module 1 is complete.**
- **Module 2 (travel filter with the LLM): working, first version done** (the user wants this module done fast, no slow step-by-step). `src/llm/ollama_client.py`: `chat(model, messages, temperature, max_tokens, think, keep_alive, seed, format)` -> (text, seconds), shared by all LLM tasks; it **auto-starts `ollama serve`** (detached) if the server is down; `format` = a JSON schema that Ollama enforces. `src/travel_filter/llm_filter.py`: `classify_travel(text, model, version=2)`. **v1** = short zero-shot prompt, one word (`PROMPT_V1`). **v2 (default)** = a stricter definition (`PROMPT_V2`) + 16 few-shot examples from **train** (`FEW_SHOT_V2`) + an answer forced by the JSON schema `{"label": "TRAVEL"|"OTHER"}`. An unclear answer counts as travel. `qwen3:4b` was tried and removed (Ollama's tag is thinking-only, ignores `think=False`). **Mistral only.** Notebook `notebooks/02_travel_filter_llm.ipynb` (executed headless with nbconvert, which is in requirements): val results v1 → v2: accuracy 88.7% → **93.4%**, F1(non-travel) 64.2% → **80.5%**, FRR 7.4% → **6.0%** (35 refused), unclear 14 → **0**, speed 0.73 → **1.38 s**/message. Predictions are cached in `eval/predictions/travel_filter_mistral_val_v{1,2}.jsonl` (`RERUN=False`); metrics in `eval/metrics/travel_filter_llm.json` (final v2, includes v1) and `travel_filter_llm_v1.json`. Remaining v2 errors: travel refused = mostly **currency conversion** + payment; non-travel let through = Tunisian 'fama ___ behi fi <city>?'. Open options for the user: more currency/payment few-shot examples (v3), a currency-lexicon rule → TRAVEL, and speed (1.4 s filter + answer LLM vs the 2 s target).

- **Module 3 (MLP on TF-IDF): done (first version).** `notebooks/03_classifier_mlp_tfidf.ipynb` (executed headless; it takes ~15 min, mostly the grid; nbconvert needs `--ExecutePreprocessor.timeout=-1`). First MLP (words, 256 neurons, 40 epochs): val 88.8%, train 100% → diagnosed as shortcut learning from templated data (1,524 template rows from 575 templates; skeletons like 'bring … to <loc>' → weather; 36–44% of val messages have unseen words; 41% of errors inside the same category). Grid (64 neurons; copies all/1 per template × words / words+pairs / words+chars × alpha 1e-4/1e-3/1e-2, early stopping max 20 epochs / patience 3, + the best with 256). **Final: words + char_wb 2–5 TF-IDF (FeatureUnion), 64 hidden, alpha 0.0001, 20 epochs, all training rows: val accuracy 93.1%, macro F1 93.2%, category accuracy 97.1%, en 91.1 / fr 92.9 / tn 95.4.** Char pieces were the key (+4–5 pts); word pairs, one copy per template, stronger L2 and 256 neurons did not help. Main remaining error: flight_price ↔ flight_search (13/40); a labelling rule for price vs search is an open user decision. Saved `models/classifier/mlp_tfidf.joblib` (vectorizer + MLP + settings) and `eval/metrics/classifier_mlp_tfidf.json` (+ the grid); figures in `eval/figures/`. `src/classifier/classify.py`: `classify(text_masked) -> {subcategory, category, confidence, top2}` (top2 lets the orchestrator ask for clarification on near ties). scikit-learn's MLP is CPU-only, single core; a GPU would need a PyTorch CUDA build (~2.5 GB download), not installed.
- **Module 4 (NER, XLM-R on Colab): model trained.** The user ran `notebooks/04_ner_xlmr_colab.ipynb` on a Colab T4 (transformers 5.x: `warmup_ratio` no longer exists → `warmup_steps=0.1`). Model unzipped into **`models/ner_xlmr/`** (546 MB, fp16, gitignored; contains `metrics.json` + learning-curve png). Results: train F1 99.9%, **val entity F1 99.2%** (LOC 99.6, DATE 97.1; en 99.3 / fr 99.8 / tn 98.5) → good fit; **challenge set F1 94.1%** (LOC 99.3% on unseen places thanks to pretraining, DATE 85.7%: misses extended/unknown date phrases like 'tomorrow morning', 'tmrw', 'ba3d 3 iyem', 'el chta hedhi'). Several val 'mistakes' are label gaps in our data (on Sundays, les prochains jours, jem3a ejjeya); real errors: TND → LOC, jwan → LOC, jedda → DATE. `src/ner/extract.py`: `extract_entities(text) -> [{text, label, start, end}]` (lru_cache load ~7 s once, then ~45–70 ms per message on CPU; same word splitting as training). Next: normalization and rules (fuzzy place matching: accept ≥ ~0.85, ask between ~0.75 and 0.85, only on NER spans; date normalization to ISO incl. phrases the model finds but the lexicon lacks; currencies/amounts/counts/airlines/origin vs destination by rules; no slang list; ambiguous currency → country of the place, else default).

Note for Module 5: there is no mock data for `activities`, `visa_documents`, `payment_methods`, `public_transport`, `car_rental` or `transfer_taxi` (the plan only lists weather, exchange, flights and hotels). Decide with the user whether to add small static info files or let the LLM answer generally with a disclaimer.

Extra fields beyond the plan's schema: each entity has `norm` (canonical city, or a date rule like `+1d`, `weekday:4`, `md:12-20`, `month:10`), used to evaluate Module 4 normalization. Each row has `template_id`, which can be used for a template-aware split so test sentences don't share templates with train (decide with the user at the split step).

## Conventions

- Run scripts as modules from the repo root with the venv's Python: `C:\Users\medbe\venvs\travel-chatbot\Scripts\python.exe -m src.data_gen.fill_templates`. Imports are `from src.labels import ...`.
- Put temporary check scripts in the session scratchpad, not in the repo.
- **Training happens in notebooks (decided by the user):** each module's training, metrics, graphs and error analysis live in a local Jupyter notebook run in VS Code (`notebooks/NN_<module>.ipynb`), built **one or two cells per turn**, with plain-language markdown before each code cell. The user runs the cells and asks questions before the next ones are added. Reusable code (`normalize()`, data prep, `detect_language()`…) stays in `src/` and the notebook imports it; the trained model is saved to `models/<module>/`. Before trusting final numbers, "Restart & Run All". The user wants training and metrics explained slowly, assuming no prior knowledge.
