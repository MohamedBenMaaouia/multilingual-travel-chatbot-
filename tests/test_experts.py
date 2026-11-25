"""UNIT TESTS - the 5 travel experts (src/pipeline/experts.py).

For each of the 15 subcategories: given the slots in memory, does the expert pick the right tool, pass the right
arguments, and fill the right defaults (origin Tunis, 1 night, today...)? The tools are real (they only read
files), so this also checks that every subcategory gets real data back. No model, no LLM.
"""
from types import SimpleNamespace

import pytest

from src.pipeline.experts import EXPERTS, estimate_trip_cost, plan_and_call, run_expert
from src.pipeline.state import Context

from conftest import TODAY      # 2026-11-25

DJERBA = {"place": "Djerba", "place_kind": "city"}

CASES = [   # topic, slots in memory, expected tool, expected arguments, expected defaults
    ("flight_search", {"destination": "Paris"}, "search_flights",
     {"origin": "Tunis", "destination": "Paris"}, {"origin": "Tunis", "date": "none: weekly timetable"}),
    ("flight_price", {"origin": "Djerba", "destination": "Paris", "date": "2026-11-27"}, "search_flights",
     {"origin": "Djerba", "destination": "Paris", "date": "2026-11-27"}, {}),
    ("baggage", {"airline": "Tunisair"}, "get_baggage_rules", {"airline": "Tunisair"}, {}),
    ("hotel_search", {**DJERBA, "date": "2026-11-27"}, "search_hotels",
     {"city": "Djerba", "checkin": "2026-11-27", "checkout": "2026-11-28"}, {"nights": 1}),
    ("hotel_price", {**DJERBA, "date": "2026-11-27", "nights": 3, "amount": 100, "from_currency": "EUR"},
     "search_hotels",
     {"city": "Djerba", "checkin": "2026-11-27", "checkout": "2026-11-30", "max_price": 340.0}, {}),   # 100 EUR
    ("hotel_facilities", DJERBA, "search_hotels",
     {"city": "Djerba", "checkin": "2026-11-25", "checkout": "2026-11-26"}, {"date": "2026-11-25", "nights": 1}),
    ("weather", DJERBA, "get_weather", {"city": "Djerba", "date": "2026-11-25"}, {"date": "2026-11-25"}),
    ("activities", DJERBA, "get_activities", {"city": "Djerba"}, {}),
    ("visa_documents", {"place": "Schengen area", "place_kind": "region"}, "get_visa_info",
     {"destination": "Schengen area"}, {}),
    ("exchange_rate", {"from_currency": "EUR", "amount": 200}, "get_exchange_rate",
     {"from_currency": "EUR", "to_currency": "TND", "amount": 200}, {"to_currency": "TND"}),
    ("exchange_rate", {"from_currency": "TND"}, "get_exchange_rate",                 # from dinars: default = euros
     {"from_currency": "TND", "to_currency": "EUR"}, {"to_currency": "EUR"}),
    ("trip_cost", {"place": "Istanbul", "place_kind": "city", "people": 2}, "estimate_trip_cost",
     {"city": "Istanbul", "origin": "Tunis", "start": "2026-12-25", "nights": 7, "people": 2},
     {"origin": "Tunis", "date": "2026-12-25", "nights": 7}),
    ("payment_methods", {"place": "Turkey", "place_kind": "country"}, "get_payment_info", {"place": "Turkey"}, {}),
    ("car_rental", {"place": "Italy", "place_kind": "country"}, "get_car_rental", {"place": "Italy"}, {}),
    ("transfer_taxi", {"place": "Paris", "place_kind": "city"}, "get_airport_transfer", {"city": "Paris"}, {}),
    ("public_transport", {"place": "Tunis", "place_kind": "city"}, "get_public_transport", {"city": "Tunis"}, {}),
]


@pytest.mark.parametrize("topic, slots, tool, args, defaults", CASES, ids=[c[0] for c in CASES])
def test_each_subcategory_calls_the_right_tool_with_the_right_defaults(topic, slots, tool, args, defaults):
    got_tool, got_args, got_defaults, data = plan_and_call(topic, slots, TODAY)
    assert (got_tool, got_args, got_defaults) == (tool, args, defaults)
    assert data["found"] is True                       # every case above has real mock data


def test_every_subcategory_is_covered():
    assert len({c[0] for c in CASES}) == 15


def test_trip_cost_adds_return_flights_per_person_and_one_room_per_two_people():
    r = estimate_trip_cost("Istanbul", "Tunis", "2026-12-25", 7, 3)
    assert r["hotel"]["rooms"] == 2                    # 3 people -> 2 rooms
    assert r["total_tnd"] == r["flight_return_per_person_tnd"] * 3 + r["hotel"]["total_tnd"]


def test_missing_information_after_two_questions_gives_a_not_found_result():
    runtime = SimpleNamespace(context=Context(today=TODAY))
    out = run_expert("hotels", {"topic": "hotel_search", "slots": DJERBA}, runtime)     # no date
    assert out["result"]["data"] == {"found": False, "reason": "missing information: date"}


def test_the_result_is_written_for_the_responder_and_kept_in_memory():
    runtime = SimpleNamespace(context=Context(today=TODAY))
    out = EXPERTS["destination"]({"topic": "weather", "slots": DJERBA}, runtime)
    assert out["result"] == out["last_result"]
    assert set(out["result"]) == {"topic", "tool", "arguments", "defaults_used", "data"}
    assert "destination_expert" in out["timings"]


def test_an_expert_refuses_a_topic_of_another_category():
    runtime = SimpleNamespace(context=Context(today=TODAY))
    with pytest.raises(AssertionError):
        EXPERTS["money"]({"topic": "weather", "slots": DJERBA}, runtime)
