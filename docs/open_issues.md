# Open issues (to revisit at the end of the project)

## 1. The travel filter judges each message alone

**Problem.** The travel filter (Mistral, Module 2, prompt v3) sees only the current message. Short messages that
continue the conversation make no sense alone, so it labels them OTHER:

| Conversation | Message | Filter (alone) | Should be |
|---|---|---|---|
| after "vol tunis paris ghodwa" | "la, ba3d ghodwa" (correction) | OTHER | travel (change the date) |
| after "nheb billet l paris" | "ghodwa" (a date) | OTHER | travel (date of the flight) |

**Temporary fix (in place now).** In the Supervisor, rules A (answer to our question) and B (follow-up/correction)
do not look at the filter: a topic in memory + a message bringing new values (place, date, amount...) = follow-up,
whatever the filter says.

**Weakness of the temporary fix (raised by the user).** A message can bring a value without being a follow-up,
e.g. "a3tini recette mlou5iya mta3 sfax" after a weather question: it contains a place, so it would be treated
as a weather follow-up for Sfax.

**User's suggested solution.** Give the filter the context: append the previous messages (or the last
question/answer) to the current one, so the LLM judges "la, ba3d ghodwa" knowing it follows a flight search.
Things to look at when we do it: how many previous messages to append, the extra time per call (longer prompt),
re-measuring on val (val messages have no context, so a small set of multi-turn conversations is also needed),
and whether the Supervisor's rule B can then rely on the filter again.

## 2. Filter error on a full travel question

"ta9es f fransa?" (weather in France) and "ta9es f touens?" were labelled OTHER by the filter (prompt v3): the
filter seems to miss the Tunisian weather word "ta9es". With no topic in memory, no rule can catch it. Not fixed.
To check when the filter is revisited (issue 1) or re-measured on val.

## 4. Tunisian greetings get a French reply (first message)

fastText does not know greeting words (whole-word model, greetings were not in the training data): "ya3tik saha",
"bonjour" and "hello" all give the same default guess, fr 62%. On a first message the Chat agent then tells Mistral
"reply in the user's language", and Mistral does not recognise "salam", "aslema", "ya3tik saha" as Tunisian, so it
replies in French. English and French greetings are fine. Idea not yet tested: "if the message is Tunisian or you
are not sure, reply in Tunisian Arabizi". The user decided to look into it later (model problem).

## 5. Responder: weak Tunisian answers

English and French answers are good (facts right, defaults mentioned, "not found" explained). Tunisian answers:
- Mistral copies the few-shot examples word for word: an exchange-rate answer ended with the weather example's
  sentence "Hedha mo3adel mta3 el chhar, mouch météo mta3 nhar bidou" and called a price "s5oun" (hot);
- lists switch to English ("stars, night, total, half board, rating") and the answer starts by repeating the request.
Ideas: (a) remove the topic-specific sentence from the weather example (the disclaimer is already in the
instructions); (b) add a Tunisian hotel-list example (to be approved by the user); (c) the QLoRA idea (a small model
fine-tuned on approved Tunisian answers). The user chose to improve this at the end.

## 6. Speed

Measured end to end (warm): clarification ~2 s; short answers ~9–11 s; long lists 25–45 s (first words after
3.5–5.5 s, streamed). The brief's target is a full response in < 2 s: not reachable with Mistral 7B on a 4 GB GPU
(8.6 tokens/s). The user accepts long answers; streaming makes the wait visible.

## 3. Filter prompt v3 not re-measured on val

v3 (v2 + "greetings, thanks and acknowledgements are OTHER") was only checked on 23 hand-picked messages
(small talk 14/15, short travel questions 8/8). "behi" is still labelled TRAVEL. The user chose not to re-measure
on val for now.
