"""Stage 3 - the 5 travel experts: Flights, Hotels, Destination, Money, Local transport.

An expert READS `topic` + `slots` (already checked by the Supervisor), and does 4 things:
    1. defaults      fill optional information nobody gave   (origin = Tunis, nights = 1, date = today...)
    2. pick the tool a fixed choice per subcategory            (flight_price -> search_flights)
    3. map the names slot names -> the tool's argument names  (slot "place" -> argument "city")
    4. call the tool and WRITE `result` for the Responder:
         {"topic", "tool", "arguments", "defaults_used", "data": <the tool's answer>}
No LLM, no model: plain Python, about a millisecond. "Not found" is a normal result (data.found = False):
the Responder explains it to the user.

The 5 experts follow the same steps, so they share `run_expert()`; each one is still its own node in the graph
and only accepts its own subcategories.
"""
import time
from datetime import date, timedelta

from langgraph.runtime import Runtime

from src.labels import TAXONOMY
from src.mock_api.service import (get_activities, get_airport_transfer, get_baggage_rules, get_car_rental,
                                  get_exchange_rate, get_payment_info, get_public_transport, get_visa_info,
                                  get_weather, search_flights, search_hotels)
from src.pipeline.state import ChatState, Context

DEFAULT_ORIGIN = "Tunis"      # our users are in Tunisia
DEFAULT_NIGHTS = 1            # hotels: price for one night if the user gave no length
TRIP_NIGHTS = 7               # trip_cost: a one-week trip if the user gave no length
TRIP_DAYS_AHEAD = 30          # trip_cost: in a month if the user gave no date


def _default(slots, key, value, used):
    """The slot if the user gave it, otherwise the default (and remember that a default was used)."""
    if key in slots:
        return slots[key]
    used[key] = value
    return value


def _plus_days(iso, n):
    return (date.fromisoformat(iso) + timedelta(days=n)).isoformat()


def _cheapest_flight(origin, destination, day):
    """Cheapest price that day; if nothing flies that day, the route's lowest price; None if no route."""
    r = search_flights(origin, destination, day)
    if r.get("found"):
        return min(f["price_tnd"] for f in r["flights"])
    t = search_flights(origin, destination)
    return min(f["price_from_tnd"] for f in t["timetable"]) if t.get("found") else None


def estimate_trip_cost(city, origin, start, nights, people):
    """trip_cost combines two tools: return flight (per person) + the cheapest hotel (one room per 2 people)."""
    end = _plus_days(start, nights)
    go, back = _cheapest_flight(origin, city, start), _cheapest_flight(city, origin, end)
    hotels = search_hotels(city, start, end)
    if not hotels.get("found") and go is None:
        return {"found": False, "reason": hotels.get("reason", "no data for this city")}
    hotel = hotels["hotels"][0] if hotels.get("found") else None
    rooms = (people + 1) // 2
    flights_total = (go + back) * people if go is not None and back is not None else 0
    hotel_total = hotel["total_tnd"] * rooms if hotel else 0
    return {"found": True, "city": city, "origin": origin, "start": start, "end": end, "nights": nights,
            "people": people,
            "flight_return_per_person_tnd": (go + back) if flights_total else None,
            "hotel": {"name": hotel["name"], "stars": hotel["stars"], "rooms": rooms,
                      "total_tnd": hotel_total} if hotel else None,
            "total_tnd": flights_total + hotel_total,
            "note": "estimate: cheapest flights + cheapest hotel; food, transport and activities not included"}


def plan_and_call(topic, slots, today):
    """Steps 1-3 (defaults, tool, argument names) and step 4 (the call). Returns (tool, arguments, used, data)."""
    used = {}
    s = slots

    # ----- Flights expert -----
    if topic in ("flight_search", "flight_price"):
        args = {"origin": _default(s, "origin", DEFAULT_ORIGIN, used), "destination": s["destination"]}
        if "date" in s:
            args["date"] = s["date"]
        else:
            used["date"] = "none: weekly timetable"
        return "search_flights", args, used, search_flights(**args)
    if topic == "baggage":
        args = {"airline": s["airline"]}
        return "get_baggage_rules", args, used, get_baggage_rules(**args)

    # ----- Hotels expert -----
    if topic in ("hotel_search", "hotel_price", "hotel_facilities"):
        checkin = s["date"] if topic != "hotel_facilities" else _default(s, "date", today.isoformat(), used)
        checkout = s.get("checkout") or _plus_days(checkin, _default(s, "nights", DEFAULT_NIGHTS, used))
        args = {"city": s["place"], "checkin": checkin, "checkout": checkout}
        if topic != "hotel_facilities" and "amount" in s:          # "hotel f jerba b 200 dt" -> budget per night
            amount, cur = s["amount"], s.get("from_currency", "TND")
            args["max_price"] = amount if cur == "TND" else get_exchange_rate(cur, "TND", amount).get("converted")
        return "search_hotels", args, used, search_hotels(**args)

    # ----- Destination expert -----
    if topic == "weather":
        args = {"city": s["place"], "date": _default(s, "date", today.isoformat(), used)}
        return "get_weather", args, used, get_weather(**args)
    if topic == "activities":
        args = {"city": s["place"]}
        return "get_activities", args, used, get_activities(**args)
    if topic == "visa_documents":
        args = {"destination": s["place"]}
        return "get_visa_info", args, used, get_visa_info(**args)

    # ----- Money expert -----
    if topic == "exchange_rate":
        frm = s["from_currency"]
        args = {"from_currency": frm, "to_currency": _default(s, "to_currency", "TND" if frm != "TND" else "EUR", used)}
        if "amount" in s:
            args["amount"] = s["amount"]
        return "get_exchange_rate", args, used, get_exchange_rate(**args)
    if topic == "payment_methods":
        args = {"place": s["place"]}
        return "get_payment_info", args, used, get_payment_info(**args)
    if topic == "trip_cost":
        args = {"city": s["place"], "origin": _default(s, "origin", DEFAULT_ORIGIN, used),
                "start": _default(s, "date", (today + timedelta(days=TRIP_DAYS_AHEAD)).isoformat(), used),
                "nights": _default(s, "nights", TRIP_NIGHTS, used), "people": _default(s, "people", 1, used)}
        return "estimate_trip_cost", args, used, estimate_trip_cost(**args)

    # ----- Local transport expert -----
    if topic == "car_rental":
        args = {"place": s["place"]}
        return "get_car_rental", args, used, get_car_rental(**args)
    if topic == "transfer_taxi":
        args = {"city": s["place"]}
        return "get_airport_transfer", args, used, get_airport_transfer(**args)
    if topic == "public_transport":
        args = {"city": s["place"]}
        return "get_public_transport", args, used, get_public_transport(**args)

    raise ValueError(f"unknown topic {topic}")


def run_expert(name, state, runtime):
    t = time.perf_counter()
    topic = state["topic"]
    assert topic in TAXONOMY[CATEGORY_OF_EXPERT[name]], f"{name} expert got topic {topic}"
    try:
        tool, args, used, data = plan_and_call(topic, state.get("slots") or {}, runtime.context.today)
    except KeyError as missing:            # the Supervisor stopped asking after 2 questions: answer that it's missing
        tool, args, used, data = None, {}, {}, {"found": False, "reason": f"missing information: {missing.args[0]}"}
    result = {"topic": topic, "tool": tool, "arguments": args, "defaults_used": used, "data": data}
    return {"result": result, "last_result": result, "timings": {f"{name}_expert": time.perf_counter() - t}}


CATEGORY_OF_EXPERT = {"flights": "flights", "hotels": "accommodation", "destination": "destination_info",
                      "money": "budget_money", "local_transport": "local_transport"}


def flights_expert(state: ChatState, runtime: Runtime[Context]):
    return run_expert("flights", state, runtime)


def hotels_expert(state: ChatState, runtime: Runtime[Context]):
    return run_expert("hotels", state, runtime)


def destination_expert(state: ChatState, runtime: Runtime[Context]):
    return run_expert("destination", state, runtime)


def money_expert(state: ChatState, runtime: Runtime[Context]):
    return run_expert("money", state, runtime)


def local_transport_expert(state: ChatState, runtime: Runtime[Context]):
    return run_expert("local_transport", state, runtime)


EXPERTS = {"flights": flights_expert, "hotels": hotels_expert, "destination": destination_expert,
           "money": money_expert, "local_transport": local_transport_expert}
