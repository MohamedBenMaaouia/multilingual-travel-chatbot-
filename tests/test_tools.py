"""UNIT TESTS - the travel data tools (src/mock_api/service.py).

Each test calls one tool with a known input and checks the answer against values we know are in data/mock_api/.
No model, no LLM: the whole file runs in about a second.
If someone edits the mock data files, these tests show exactly which answers changed.
"""
from src.mock_api.service import (get_activities, get_airport_transfer, get_baggage_rules, get_car_rental,
                                  get_exchange_rate, get_payment_info, get_public_transport, get_visa_info,
                                  get_weather, search_flights, search_hotels)


# ---------- weather ----------

def test_weather_djerba_july_is_hot_and_dry():
    w = get_weather("Djerba", "2026-07-15")
    assert w["found"]
    assert (w["temp_min"], w["temp_max"], w["rain_days"], w["sea_temp"]) == (24, 33, 0, 26)
    assert "not a day forecast" in w["note"]


def test_weather_tozeur_january_has_cold_nights_and_no_sea():
    w = get_weather("Tozeur", "2026-01-15")
    assert w["temp_min"] <= 5
    assert w["sea_temp"] is None                      # Tozeur is in the desert


def test_weather_city_name_ignores_case():
    assert get_weather("djerba", "2026-07-15")["temp_max"] == 33


def test_weather_unknown_city_is_not_found_with_a_reason():
    w = get_weather("Atlantis", "2026-01-15")
    assert w["found"] is False and w["reason"]


# ---------- exchange rates ----------

def test_exchange_euro_to_dinar_with_amount():
    r = get_exchange_rate("EUR", "TND", 200)
    assert r["rate"] == 3.4 and r["converted"] == 680.0


def test_exchange_works_in_both_directions():
    there = get_exchange_rate("EUR", "TND")["rate"]
    back = get_exchange_rate("TND", "EUR")["rate"]
    assert abs(there * back - 1) < 0.001


def test_exchange_unknown_currency_lists_the_known_ones():
    r = get_exchange_rate("XYZ", "TND")
    assert r["found"] is False and "XYZ" in r["reason"]
    assert "EUR" in r["known_currencies"]


# ---------- flights ----------

def test_flights_tunis_paris_on_a_friday_cheapest_first():
    r = search_flights("Tunis", "Paris", "2026-11-27")
    assert r["found"]
    prices = [f["price_tnd"] for f in r["flights"]]
    assert prices == sorted(prices)
    assert r["flights"][0]["airline"] == "Transavia" and prices[0] == 415


def test_flights_cost_25_percent_more_in_july_and_august():
    november = search_flights("Tunis", "Paris", "2026-11-27")["flights"][0]["price_tnd"]
    july = search_flights("Tunis", "Paris", "2027-07-02")["flights"][0]["price_tnd"]     # also a Friday
    assert (november, july) == (415, 519)                                               # 415 x 1.25 = 518.75


def test_flights_without_a_date_give_the_weekly_timetable():
    r = search_flights("Tunis", "Paris")
    assert r["found"] and "timetable" in r and "flights" not in r
    assert {"days", "departure", "season", "price_from_tnd"} <= set(r["timetable"][0])


def test_flights_summer_only_route_explains_why_in_november():
    r = search_flights("Djerba", "Brussels", "2026-11-27")
    assert r["found"] is False and "only flies in summer" in r["reason"]


def test_flights_same_airport_needs_no_flight():
    r = search_flights("Tunis", "Tunis", "2026-11-27")
    assert r["found"] is False and "same airport" in r["reason"]


def test_flights_unknown_city_has_no_airport():
    assert "no airport known" in search_flights("Tunis", "Atlantis", "2026-11-27")["reason"]


def test_baggage_rules_of_tunisair():
    r = get_baggage_rules("Tunisair")
    assert (r["cabin_bag_kg"], r["checked_bag_kg"], r["extra_bag_price_tnd"]) == (8, 23, 150)


def test_baggage_unknown_airline_lists_the_known_ones():
    r = get_baggage_rules("FooAir")
    assert r["found"] is False and "Tunisair" in r["known_airlines"]


# ---------- hotels ----------

def test_hotels_are_sorted_and_totals_match_the_nights():
    r = search_hotels("Hammamet", "2026-06-29", "2026-07-02")
    assert r["found"] and r["nights"] == 3
    prices = [h["price_per_night_tnd"] for h in r["hotels"]]
    assert prices == sorted(prices)
    first = r["hotels"][0]
    assert abs(first["total_tnd"] - 3 * first["price_per_night_tnd"]) <= 2      # per-night price is rounded


def test_hotels_too_low_budget_gives_the_cheapest_price():
    r = search_hotels("Hammamet", "2026-06-29", "2026-07-02", max_price=50)
    assert r["found"] is False and r["cheapest_tnd"] == 145


def test_hotels_checkout_must_be_after_checkin():
    r = search_hotels("Hammamet", "2026-07-02", "2026-07-02")
    assert r["found"] is False and "after checkin" in r["reason"]


# ---------- visa, payment, car rental (country information) ----------

def test_visa_for_a_city_uses_its_country():
    assert get_visa_info("Paris")["country"] == "France"
    assert get_visa_info("Paris")["type"] == get_visa_info("France")["type"]


def test_visa_for_europe_is_the_common_schengen_visa():
    r = get_visa_info("Schengen area")
    assert r["found"] and "Schengen" in r["type"] and "France" in r["countries"]
    assert r["passport"] == "Tunisian"


def test_visa_for_the_gulf_gives_one_summary_per_country():
    r = get_visa_info("Gulf")
    assert [c["country"] for c in r["by_country"]] == ["UAE", "Qatar", "Saudi Arabia"]


def test_payment_info_for_istanbul_is_about_turkey():
    r = get_payment_info("Istanbul")
    assert r["country"] == "Turkey" and r["currency"] == "TRY"


def test_car_rental_in_a_region_asks_which_country():
    r = get_car_rental("Schengen area")
    assert r["found"] is False and "which country" in r["reason"]


# ---------- city information ----------

def test_activities_transfer_and_public_transport():
    assert "Houmt Souk market" in get_activities("Djerba")["activities"]
    assert get_airport_transfer("Paris")["taxi_tnd"] == [190, 220]
    assert get_public_transport("Tunis")["ticket_tnd"] == 1


def test_city_information_unknown_city():
    assert get_activities("Atlantis")["found"] is False
