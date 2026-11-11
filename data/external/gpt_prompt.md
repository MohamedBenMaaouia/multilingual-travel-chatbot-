# Prompt for an external model (GPT) to generate extra queries

How to use:
1. Paste **PART 1** once at the start of a new chat.
2. Then send the **PART 2** requests one by one (21 messages). After each answer, copy the JSONL lines
   (only the lines, no ``` fences) and append them to `data/external/gpt_queries.jsonl`.
3. Bring the file back to Claude Code for labeling, cleaning and merging.

---

## PART 1 (paste once)

You are helping me build a training dataset for a travel-agency chatbot (university AI project).
I need realistic messages that customers would type into the chat. You will generate them in batches
when I ask. Read all the rules carefully; I will process your output automatically.

### Languages
- `en`: English, typed in a chat window. Mix of casual and polite. Short, like real chat messages.
- `fr`: French, typed by a French or Tunisian person. Mix of formal and informal; sometimes no accents,
  sometimes no question mark.
- `tn_latin`: **Tunisian Arabic dialect (Derja / Tounsi) written in Latin letters (Arabizi)**, exactly as
  Tunisians type on WhatsApp/Facebook. Digits stand for Arabic sounds: 3=ع, 7=ح, 9=ق, 5=خ, 8=غ.
  Tunisians naturally mix in French words (billet, hotel, location, voyage, reservation, valise...).
  Typical words: nheb, fama, 9adech, b9adech, chnowa, chnouma, kifech, win, najem, lazem, lazemni, barcha,
  behi, m3a, mta3, mte3i, karhba, tayara, matar, flous, nkhalles, nsafer, nemchi.
  Mostly lowercase. **Never** write Modern Standard Arabic, Moroccan, Algerian or Egyptian dialect.
  **Never** use Arabic script.

### Travel subcategories (the topic of each message)
| subcategory | meaning |
|---|---|
| flight_search | finding available flights: routes, schedules, airlines, direct or not (NOT price) |
| flight_price | price/cost of plane tickets, cheap flights, fares |
| baggage | luggage rules on flights: allowance, weight, cabin bag, extra suitcase, what can be carried |
| hotel_search | finding a hotel/place to stay, availability, recommendations (NOT price) |
| hotel_price | price of hotels, nights, rooms |
| hotel_facilities | what a hotel offers: pool, wifi, breakfast, parking, spa, pets, accessibility |
| car_rental | renting a car: price, conditions, documents, pick-up/drop-off |
| transfer_taxi | taxis, airport transfers, shuttles, private drivers, ride apps |
| public_transport | trains, buses, louages (Tunisian shared taxis), metro, ferries, timetables |
| weather | weather, temperature, rain, wind, what to wear because of the climate |
| activities | things to do and see: sights, beaches, excursions, restaurants, nightlife |
| visa_documents | visas, passports, documents needed to enter a foreign country |
| exchange_rate | currency exchange rates, converting amounts, where to change money |
| trip_cost | total budget or cost of a trip or stay |
| payment_methods | how to pay: card, cash, ATMs, bank transfer, installments (information only) |

### STRICT rule 1: cities
If a message mentions a city, use ONLY these exact spellings (for the message's language).
Never mention any other city, country, region, airport name or landmark (no "France", "Tunisia", "Europe",
"Carthage", "Nice", "Cairo"...).
- For `visa_documents`, use only the foreign cities (second group).
- For `public_transport`, use only the Tunisian cities (first group).

**en:** Tunisian: Tunis, Djerba, Jerba, Sousse, Hammamet, Monastir, Mahdia, Sfax, Tozeur, Tabarka, Bizerte,
Nabeul, Kairouan, Gabes, Tataouine, Douz, Kelibia, Sidi Bou Said, Ain Draham, Matmata, Zarzis.
Foreign: Paris, Marseille, Lyon, Rome, Milan, Istanbul, Dubai, London, Madrid, Barcelona, Berlin, Frankfurt,
Brussels, Geneva, Amsterdam, Lisbon, Malta, Algiers, Casablanca, Jeddah, Doha, Montreal, New York.

**fr:** Tunisian: Tunis, Djerba, Sousse, Hammamet, Monastir, Mahdia, Sfax, Tozeur, Tabarka, Bizerte, Nabeul,
Kairouan, Gabès, Tataouine, Douz, Kélibia, Sidi Bou Saïd, Aïn Draham, Matmata, Zarzis.
Foreign: Paris, Marseille, Lyon, Rome, Milan, Istanbul, Dubaï, Londres, Madrid, Barcelone, Berlin, Francfort,
Bruxelles, Genève, Amsterdam, Lisbonne, Malte, Alger, Casablanca, Djeddah, Doha, Montréal, New York.

**tn_latin:** Tunisian: tunis, tounes, djerba, jerba, jarba, sousse, soussa, hammamet, hamamet, monastir, mestir,
mahdia, mehdia, sfax, sfa9s, tozeur, touzer, tabarka, tbarka, bizerte, bnzart, benzart, nabeul, nabel, kairouan,
9irwan, gabes, 9abes, tataouine, tatawin, douz, kelibia, klibia, sidi bou said, ain draham, 3in draham, matmata,
zarzis, jarjis.
Foreign: paris, marseille, marsilia, lyon, roma, milano, istanbul, stanbul, dubai, london, londres, madrid,
barcelona, berlin, frankfurt, bruxelles, geneve, amsterdam, lisbonne, malta, alger, dzayer, casablanca, casa,
jeddah, doha, montreal, new york.

### STRICT rule 2: dates
If a message mentions a date or time, use ONLY these expressions. Never use any other date word, holiday,
season, "in 3 days", "this week", etc.

**en:** today, tonight, tomorrow, the day after tomorrow, this weekend, next weekend, next week, in two weeks,
next month, this summer, at the end of the month, next Monday / on Monday (any weekday),
on December 20 / on 20 December / on the 20th of December (any day + month), in December (any month).

**fr:** aujourd'hui, ce soir, demain, après-demain, ce week-end, le week-end prochain, la semaine prochaine,
dans deux semaines, le mois prochain, cet été, à la fin du mois, lundi prochain / lundi (any weekday),
le 20 décembre / le 20/12 (any day + month), en décembre / début décembre (any month).

**tn_latin:** lyoum, el youm, ellila, ghodwa, 8odwa, ba3d ghodwa, fel weekend, el weekend hedha, el weekend ejjey,
el jem3a ejjeya, jem3a jeya, ba3d jem3tin, echhar ejjey, chhar jey, fel sif, a5er echhar;
weekdays as "nhar el khmis" or "el khmis ejjey" (weekdays: el tnin, ettleth, larb3a, el khmis, el jem3a, essebt,
el a7ad); "nhar 20 decembre", "20 decembre", "fi 20/12", "fi chhar 12", "fi decembre"
(months: janvier, fevrier, mars, avril, mai, juin, juillet, aout, septembre, octobre, novembre, decembre).

### Rule 3: variety (very important)
- About half of the messages mention a city, about a third mention a date, some mention neither.
- Every message must have a DIFFERENT structure. Do not start many messages with the same word.
- Mix full questions, requests, very short keyword-style messages ("prix taxi sousse"), messages with a bit
  of context ("we're 4 adults and a baby..."), polite and casual, some without a question mark.
- 3 to 30 words per message. One topic per message (the requested subcategory only).
- Amounts and numbers are free (e.g. "200 euros", "3 nuits", "23 kilo").

### Non-travel messages
When I ask for non-travel messages, write things a person might type to a chatbot that are NOT about travel:
cooking, sport/football, school/exams, health, tech/phones, jobs, shopping, family, religion, music, films,
general knowledge, jokes, greetings or small talk... About 20% should be **tricky**: they mention a city from
the lists or a date from the lists but are still NOT about travel (e.g. "who won the match in Sousse
yesterday?" is not allowed because of "yesterday", but "who won the match in Sousse?" is fine).
The same city and date rules apply.

### Output format (STRICT)
Answer ONLY with JSON Lines inside one code block, one message per line, nothing else:
```
{"lang": "tn_latin", "subcategory": "baggage", "text": "..."}
```
For non-travel messages use `"subcategory": "non_travel"`.
Do not number the lines. Do not add comments. Do not repeat a message.

Reply "ready" and wait for my first request.

---

## PART 2 (send one at a time, 21 requests)

Travel, 40 messages per subcategory (120 per request):
1. `lang=en: 40 messages each for flight_search, flight_price, baggage`
2. `lang=en: 40 messages each for hotel_search, hotel_price, hotel_facilities`
3. `lang=en: 40 messages each for car_rental, transfer_taxi, public_transport`
4. `lang=en: 40 messages each for weather, activities, visa_documents`
5. `lang=en: 40 messages each for exchange_rate, trip_cost, payment_methods`
6. to 10. The same five requests with `lang=fr`
11. to 15. The same five requests with `lang=tn_latin`

Non-travel, 110 messages per request:
16. `lang=en: 110 non_travel messages`
17. `lang=en: 110 more non_travel messages, different topics and structures from before`
18. and 19. The same two requests with `lang=fr`
20. and 21. The same two requests with `lang=tn_latin`

Tip: if the answer gets cut off, say "continue from where you stopped". If the Tunisian starts to sound like
standard Arabic or French, tell it: "more natural Tunisian Derja, the way people really type on WhatsApp".
