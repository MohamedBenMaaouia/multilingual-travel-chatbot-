# Dataset balance report

Total rows: **4519**  (duplicates dropped: 0, bad spans dropped: 0)

## Rows per language and source

| lang | template (travel) | gpt travel | gpt non-travel | total |
|---|---|---|---|---|
| en | 720 | 552 | 227 | 1499 |
| fr | 717 | 563 | 234 | 1514 |
| tn_latin | 721 | 559 | 226 | 1506 |

## Language x subcategory

| subcategory | en | fr | tn_latin | total |
|---|---|---|---|---|
| flight_search | 87 | 87 | 87 | 261 |
| flight_price | 87 | 87 | 87 | 261 |
| baggage | 83 | 82 | 82 | 247 |
| hotel_search | 87 | 87 | 87 | 261 |
| hotel_price | 87 | 87 | 87 | 261 |
| hotel_facilities | 78 | 77 | 76 | 231 |
| car_rental | 87 | 87 | 87 | 261 |
| transfer_taxi | 87 | 87 | 87 | 261 |
| public_transport | 87 | 87 | 87 | 261 |
| weather | 77 | 86 | 87 | 250 |
| activities | 87 | 87 | 87 | 261 |
| visa_documents | 87 | 87 | 83 | 257 |
| exchange_rate | 87 | 87 | 87 | 261 |
| trip_cost | 84 | 85 | 87 | 256 |
| payment_methods | 80 | 80 | 82 | 242 |
| non_travel | 227 | 234 | 226 | 687 |

## Split sizes

| split | en | fr | tn_latin | total | share |
|---|---|---|---|---|---|
| train | 1049 | 1058 | 1054 | 3161 | 69.9% |
| val | 225 | 231 | 228 | 684 | 15.1% |
| test | 225 | 225 | 224 | 674 | 14.9% |

## Test rows per language x subcategory (smallest cells matter most)

| subcategory | en | fr | tn_latin |
|---|---|---|---|
| flight_search | 12 | 12 | 14 |
| flight_price | 13 | 14 | 12 |
| baggage | 12 | 12 | 12 |
| hotel_search | 14 | 13 | 12 |
| hotel_price | 13 | 13 | 14 |
| hotel_facilities | 12 | 11 | 11 |
| car_rental | 13 | 12 | 13 |
| transfer_taxi | 13 | 14 | 14 |
| public_transport | 13 | 12 | 12 |
| weather | 11 | 13 | 14 |
| activities | 12 | 14 | 13 |
| visa_documents | 13 | 14 | 12 |
| exchange_rate | 14 | 13 | 12 |
| trip_cost | 14 | 12 | 13 |
| payment_methods | 12 | 11 | 12 |
| non_travel | 34 | 35 | 34 |

## Entities

| lang | LOC | DATE | rows with ≥1 entity |
|---|---|---|---|
| en | 1251 | 288 | 1099 |
| fr | 1240 | 258 | 1078 |
| tn_latin | 1308 | 294 | 1174 |

## Leakage checks

- Templates appearing in more than one split: **0** (must be 0)
- Identical sentences appearing in more than one split: **0** (must be 0)
