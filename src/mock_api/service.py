"""Travel data tools: the functions the travel experts call to get their data.

  weather.csv          -> get_weather
  exchange_rates.json  -> get_exchange_rate
  flights.csv          -> search_flights, get_baggage_rules
  hotels.csv           -> search_hotels
  country_info.json    -> get_visa_info, get_payment_info, get_car_rental
  city_info.json       -> get_activities, get_airport_transfer, get_public_transport

They read the files in data/mock_api/ directly (no server, no internet).
The files are loaded once and kept in memory, so each call takes about a millisecond.
The names and arguments are the ones a real travel API would have, so this file
is the only thing to change if we ever plug in real data.

Every function returns a dict with "found": True/False instead of raising an
error: "no flight that day" is a normal answer the bot must be able to explain.

Cities must be canonical names from data/lexicons/places.json (e.g. "Djerba"),
which is what the Module 4 rules (src/ner/rules.py) produce.
Dates can be a datetime.date or an ISO string "YYYY-MM-DD".
"""
import csv
import json
from datetime import date, datetime, timedelta
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "mock_api"
PLACES_FILE = ROOT / "data" / "lexicons" / "places.json"

SUMMER_MONTHS = range(4, 11)   # flights marked "summer" run April -> October, "winter" ones the other months
PEAK_MONTHS = {7, 8}           # July and August: flight prices are higher
PEAK_FACTOR = 1.25             # +25% in peak months


# ---------- reading the files (once, then kept in memory) ----------

@lru_cache(maxsize=None)
def _read_csv(name):
    with open(DATA_DIR / name, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


@lru_cache(maxsize=None)
def _rates():
    with open(DATA_DIR / "exchange_rates.json", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=None)
def _places():
    with open(PLACES_FILE, encoding="utf-8") as f:
        return json.load(f)["places"]


def _to_date(d):
    """Accept a date or an ISO string and return a date."""
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return date.fromisoformat(str(d))


def _same(a, b):
    """Compare city names without caring about upper/lower case."""
    return a.casefold() == b.casefold()


# ---------- 1. weather ----------

def get_weather(city, date):
    """Climate averages of the city for the month of `date`."""
    d = _to_date(date)
    for row in _read_csv("weather.csv"):
        if _same(row["city"], city) and int(row["month"]) == d.month:
            return {
                "found": True,
                "city": row["city"],
                "date": d.isoformat(),
                "temp_min": int(row["temp_min"]),
                "temp_max": int(row["temp_max"]),
                "condition": row["condition"],          # sunny / partly_cloudy / rainy / snowy / windy_sandy
                "rain_days": int(row["rain_days"]),     # rainy days in that month
                "humidity": int(row["humidity"]),       # %
                "sea_temp": int(row["sea_temp"]) if row["sea_temp"] else None,  # None = no sea
                "note": "monthly climate average, not a day forecast",
            }
    return {"found": False, "city": city, "reason": "no weather data for this city"}


# ---------- 2. exchange rates ----------

def get_exchange_rate(from_currency, to_currency, amount=None):
    """Rate from one currency to another (both go through TND), and the converted amount if given."""
    data = _rates()
    rates = data["rates_to_tnd"]                 # value of 1 unit of each currency, in TND
    a, b = from_currency.upper(), to_currency.upper()
    missing = [c for c in (a, b) if c not in rates]
    if missing:
        return {"found": False, "reason": f"unknown currency: {', '.join(missing)}",
                "known_currencies": sorted(rates)}
    rate = rates[a] / rates[b]                   # 1 a = rate b
    result = {"found": True, "from": a, "to": b, "rate": round(rate, 4), "as_of": data["as_of"]}
    if amount is not None:
        result["amount"] = amount
        result["converted"] = round(amount * rate, 2)
    return result


# ---------- 3. flights ----------

def _airport(city):
    for name, info in _places().items():
        if _same(name, city):
            return info.get("airport")
    return None


def _flies_on(row, d):
    """Does this timetable row have a flight on date d?"""
    weekdays = {int(x) for x in row["days_of_week"].split(",")}    # 0 = Monday
    if d.weekday() not in weekdays:
        return False
    if row["season"] == "summer":
        return d.month in SUMMER_MONTHS
    if row["season"] == "winter":
        return d.month not in SUMMER_MONTHS
    return True                                                    # all_year


def _flight_info(row, d):
    dep = datetime.combine(d, datetime.strptime(row["departure_time"], "%H:%M").time())
    arr = dep + timedelta(minutes=int(row["duration_min"]))
    price = int(row["base_price_tnd"])
    if d.month in PEAK_MONTHS:
        price = round(price * PEAK_FACTOR)
    return {
        "flight_no": row["flight_no"],
        "airline": row["airline"],
        "from_airport": row["origin_name"],
        "to_airport": row["destination_name"],
        "departure": dep.strftime("%H:%M"),
        "arrival": arr.strftime("%H:%M") + (" (+1 day)" if arr.date() > d else ""),
        "duration_min": int(row["duration_min"]),
        "stops": int(row["stops"]),
        "via": row["via"] or None,
        "price_tnd": price,
        "cabin_bag_kg": int(row["cabin_bag_kg"]),
        "checked_bag_kg": int(row["checked_bag_kg"]),       # 0 = checked bag not included
        "extra_bag_price_tnd": int(row["extra_bag_price_tnd"]),
    }


DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _timetable(route, origin, destination):
    """Weekly timetable of a route (used when the user gives no date)."""
    rows = []
    for r in sorted(route, key=lambda r: r["departure_time"]):
        days = [int(x) for x in r["days_of_week"].split(",")]
        rows.append({
            "flight_no": r["flight_no"],
            "airline": r["airline"],
            "days": "every day" if len(days) == 7 else ", ".join(DAY_NAMES[i] for i in days),
            "departure": r["departure_time"],
            "duration_min": int(r["duration_min"]),
            "stops": int(r["stops"]),
            "season": {"all_year": "all year", "summer": "April to October",
                       "winter": "November to March"}[r["season"]],
            "price_from_tnd": int(r["base_price_tnd"]),
        })
    return {"origin": origin, "destination": destination, "found": True, "timetable": rows,
            "note": "weekly timetable; prices are higher in July and August"}


def search_flights(origin, destination, date=None):
    """Flights from `origin` to `destination` on `date`, cheapest first.
    If none flies that day, `next_dates` lists the next days (within 2 weeks) that have one.
    Without a date: the weekly timetable of the route (which days, what time, which season)."""
    org, dst = _airport(origin), _airport(destination)
    if not org or not dst:
        unknown = origin if not org else destination
        return {"found": False, "reason": f"no airport known for {unknown}"}
    if org == dst:
        return {"found": False, "reason": f"{origin} and {destination} use the same airport ({org}); no flight needed"}

    route = [r for r in _read_csv("flights.csv") if r["origin"] == org and r["destination"] == dst]
    if not route:
        return {"origin": origin, "destination": destination, "found": False, "reason": "no flights on this route"}
    if date is None:
        return _timetable(route, origin, destination)

    d = _to_date(date)
    base = {"origin": origin, "destination": destination, "date": d.isoformat()}

    flights = sorted((_flight_info(r, d) for r in route if _flies_on(r, d)), key=lambda f: f["price_tnd"])
    if flights:
        return {**base, "found": True, "flights": flights}

    next_dates = []
    for k in range(1, 15):
        day = d + timedelta(days=k)
        if any(_flies_on(r, day) for r in route):
            next_dates.append(day.isoformat())
            if len(next_dates) == 3:
                break
    seasons = {r["season"] for r in route}
    if not next_dates and seasons == {"summer"}:
        return {**base, "found": False, "reason": "this route only flies in summer (April to October)"}
    if not next_dates and seasons == {"winter"}:
        return {**base, "found": False, "reason": "this route only flies in winter (November to March)"}
    return {**base, "found": False, "reason": "no flight on this date", "next_dates": next_dates}


def get_baggage_rules(airline):
    """Baggage allowance of an airline (the same on all its flights in flights.csv)."""
    rows = [r for r in _read_csv("flights.csv") if _same(r["airline"], airline)]
    if not rows:
        known = sorted({r["airline"] for r in _read_csv("flights.csv")})
        return {"found": False, "airline": airline, "reason": "no data for this airline", "known_airlines": known}
    r = rows[0]
    return {"found": True, "airline": r["airline"],
            "cabin_bag_kg": int(r["cabin_bag_kg"]),
            "checked_bag_kg": int(r["checked_bag_kg"]),            # 0 = checked bag not included in the ticket
            "extra_bag_price_tnd": int(r["extra_bag_price_tnd"])}


# ---------- 4. hotels ----------

def search_hotels(city, checkin, checkout, max_price=None):
    """Hotels in `city` for the stay, cheapest first.
    Each night is priced at the low- or high-season rate of its month.
    `max_price` = maximum average price per night, in TND."""
    d_in, d_out = _to_date(checkin), _to_date(checkout)
    nights = (d_out - d_in).days
    base = {"city": city, "checkin": d_in.isoformat(), "checkout": d_out.isoformat(), "nights": nights}
    if nights < 1:
        return {**base, "found": False, "reason": "checkout must be after checkin"}

    rows = [r for r in _read_csv("hotels.csv") if _same(r["city"], city)]
    if not rows:
        return {**base, "found": False, "reason": "no hotel data for this city"}

    hotels = []
    for r in rows:
        high_months = {int(m) for m in r["high_season_months"].split(",")}
        total = 0
        for k in range(nights):
            night = d_in + timedelta(days=k)
            high = night.month in high_months
            total += int(r["price_high_season_tnd"] if high else r["price_low_season_tnd"])
        per_night = round(total / nights)
        if max_price is not None and per_night > max_price:
            continue
        hotels.append({
            "name": r["name"],
            "stars": int(r["stars"]),
            "price_per_night_tnd": per_night,
            "total_tnd": total,
            "board": r["board"],                               # room_only / bed_and_breakfast / half_board / ...
            "facilities": r["facilities"].split(";"),
            "rating": float(r["rating"]),                      # out of 10
        })

    hotels.sort(key=lambda h: h["price_per_night_tnd"])
    if not hotels:
        return {**base, "found": False, "reason": f"no hotel under {max_price} TND per night",
                "cheapest_tnd": min(int(r["price_low_season_tnd"]) for r in rows)}
    return {**base, "found": True, "hotels": hotels}


# ---------- 5-7. country information: visa, payment, car rental (country_info.json) ----------

@lru_cache(maxsize=None)
def _info(name):
    with open(DATA_DIR / name, encoding="utf-8") as f:
        return json.load(f)


def _country(place):
    """Country of a place: works with a city ("Paris" -> "France") or a country name ("France")."""
    for name, info in _places().items():
        if _same(name, place):
            return info["country"]
    for country in _info("country_info.json")["countries"]:
        if _same(country, place):
            return country
    return None


@lru_cache(maxsize=None)
def _regions():
    """Region name -> member countries (from data/lexicons/countries.json)."""
    with open(ROOT / "data" / "lexicons" / "countries.json", encoding="utf-8") as f:
        return {name: r["members"] for name, r in json.load(f)["regions"].items()}


def _region(place):
    return next((name for name in _regions() if _same(name, place)), None)


def _country_part(place, part):
    region = _region(place)
    if region:                                   # payment / car rental depend on the country: the bot must ask which one
        return {"found": False, "place": place, "reason": "this is a region, not a country: ask which country",
                "countries": _regions()[region]}
    country = _country(place)
    data = _info("country_info.json")["countries"].get(country) if country else None
    if not data:
        return {"found": False, "place": place, "reason": "no information for this place"}
    return {"found": True, "place": place, "country": country, "currency": data["currency"], **data[part]}


def get_visa_info(destination):
    """Visa rules for a TUNISIAN passport going to `destination` (city, country or region).
    Region with a common visa (Schengen area) -> that visa; other regions (Gulf) -> one summary per country."""
    region = _region(destination)
    if region:
        members = _regions()[region]
        common = _info("country_info.json")["regions"].get(region, {}).get("visa")
        if common:
            return {"found": True, "place": destination, "region": region, "countries": members,
                    **common, "passport": "Tunisian"}
        countries = _info("country_info.json")["countries"]
        summary = [{"country": c, **{k: countries[c]["visa"][k] for k in
                                      ("needed", "type", "max_stay_days", "fee_tnd", "processing_days")}}
                   for c in members]
        return {"found": True, "place": destination, "region": region, "by_country": summary, "passport": "Tunisian"}
    return {**_country_part(destination, "visa"), "passport": "Tunisian"}


def get_payment_info(place):
    """Cards, cash, ATMs and tipping habits in the country of `place`."""
    return _country_part(place, "payment")


def get_car_rental(place):
    """Car rental in the country of `place`: price per day (economy car), minimum age, licence, driving side."""
    return _country_part(place, "driving")


# ---------- 8-10. city information: activities and transport (city_info.json) ----------

def _city_part(city, part):
    for name, data in _info("city_info.json")["cities"].items():
        if _same(name, city):
            return {"found": True, "city": name, **({part: data[part]} if isinstance(data[part], list) else data[part])}
    return {"found": False, "city": city, "reason": "no information for this city"}


def get_activities(city):
    """Main things to do in `city`."""
    return _city_part(city, "activities")


def get_airport_transfer(city):
    """From the nearest airport to `city`: taxi price range, time, and a cheaper alternative if any."""
    return _city_part(city, "airport_transfer")


def get_public_transport(city):
    """Public transport in `city`: what exists and a typical ticket price."""
    return _city_part(city, "public_transport")


if __name__ == "__main__":
    # Quick demo: python -m src.mock_api.service
    from pprint import pprint
    pprint(get_weather("Djerba", "2026-07-15"))
    pprint(get_exchange_rate("EUR", "TND", 200))
    pprint(search_flights("Tunis", "Paris", "2026-11-27"))
    pprint(search_hotels("Hammamet", "2026-06-29", "2026-07-02", max_price=250))
