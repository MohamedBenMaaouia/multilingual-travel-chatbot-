"""Step 1.6 - Generate the mock "API" data the chatbot will query instead of real providers.

Outputs (data/mock_api/):
  weather.csv          monthly climate per city (temp min/max, condition, rain days, humidity, sea temp)
  exchange_rates.json  value of 1 unit of each currency in TND
  hotels.csv           fictional hotels: stars, prices (low / high season), board, facilities, rating
  flights.csv          weekly timetable: route, airline, weekdays, time, duration, price, baggage rules

Everything is MOCK data: realistic-looking, but invented. The mock service (Module 5) reads these files
through functions shaped like a real API (get_weather, search_flights...), so real providers could replace
them later without changing the rest of the chatbot.

Run from the repo root:
    python -m src.mock_api.generate_data
"""

import csv
import json
import math
import random
from pathlib import Path

from src.data_gen.fill_templates import PLACES

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "mock_api"
SEED = 42

# =========================================================================== weather
# Per city: (Jan min, Jan max, Jul min, Jul max) in °C, rain regime, (sea Jan, sea Aug) or None if inland.
# Values are approximate real climate averages.
CLIMATE = {
    "Tunis": (7, 16, 22, 33, "med", (15, 27)),
    "Djerba": (9, 17, 24, 33, "dry_med", (15, 28)),
    "Sousse": (8, 17, 22, 32, "east_med", (15, 27)),
    "Hammamet": (8, 16, 21, 32, "med", (15, 27)),
    "Monastir": (9, 17, 22, 32, "east_med", (15, 27)),
    "Mahdia": (9, 17, 22, 31, "east_med", (15, 27)),
    "Sfax": (7, 17, 22, 32, "east_med", (15, 28)),
    "Tozeur": (5, 17, 26, 41, "desert", None),
    "Tabarka": (8, 16, 21, 31, "med_wet", (14, 26)),
    "Bizerte": (8, 16, 21, 31, "med", (15, 26)),
    "Nabeul": (8, 16, 21, 32, "med", (15, 27)),
    "Kairouan": (6, 17, 23, 38, "semi_arid", None),
    "Gabes": (7, 17, 23, 33, "dry_med", (15, 28)),
    "Tataouine": (6, 17, 25, 39, "desert", None),
    "Douz": (4, 17, 25, 41, "desert", None),
    "Kelibia": (9, 16, 21, 31, "med", (15, 26)),
    "Sidi Bou Said": (8, 16, 22, 32, "med", (15, 27)),
    "Ain Draham": (3, 10, 17, 31, "wet_mountain", None),
    "Matmata": (6, 15, 24, 37, "desert", None),
    "Zarzis": (9, 17, 24, 33, "dry_med", (15, 28)),
    "Paris": (3, 7, 16, 26, "oceanic", None),
    "Marseille": (3, 12, 20, 30, "euro_med", (13, 23)),
    "Lyon": (1, 7, 17, 28, "continental_mild", None),
    "Rome": (4, 12, 19, 31, "euro_med", (14, 25)),
    "Milan": (0, 7, 18, 29, "continental_mild", None),
    "Istanbul": (4, 9, 20, 29, "euro_med_wet", (9, 25)),
    "Dubai": (14, 24, 30, 41, "gulf", (22, 33)),
    "London": (3, 8, 14, 24, "oceanic", None),
    "Madrid": (3, 10, 19, 33, "dry_continental", None),
    "Barcelona": (5, 14, 21, 29, "euro_med", (13, 26)),
    "Berlin": (-2, 3, 14, 25, "continental", None),
    "Frankfurt": (-1, 4, 14, 26, "continental", None),
    "Brussels": (1, 6, 14, 23, "oceanic", None),
    "Geneva": (-1, 5, 15, 27, "continental_mild", None),
    "Amsterdam": (1, 6, 14, 22, "oceanic", None),
    "Lisbon": (9, 15, 18, 28, "atlantic_med", (15, 20)),
    "Malta": (10, 16, 22, 31, "euro_med", (15, 27)),
    "Algiers": (6, 17, 21, 31, "med", (15, 26)),
    "Casablanca": (8, 18, 19, 26, "atlantic_med", (17, 22)),
    "Jeddah": (19, 29, 28, 39, "gulf", (24, 32)),
    "Doha": (14, 22, 31, 42, "gulf", (20, 33)),
    "Montreal": (-14, -5, 16, 27, "continental_cold", None),
    "New York": (-3, 4, 21, 29, "continental", None),
}

# Rainy days per month (Jan..Dec) and average humidity (%) for each regime
RAIN = {
    "med":              ([11, 10, 9, 7, 4, 2, 1, 1, 4, 7, 9, 11], 68),
    "med_wet":          ([14, 12, 11, 9, 6, 2, 1, 2, 6, 10, 13, 15], 72),
    "east_med":         ([7, 6, 6, 5, 3, 1, 0, 1, 4, 6, 6, 7], 66),
    "dry_med":          ([4, 3, 3, 2, 1, 0, 0, 0, 2, 4, 3, 4], 62),
    "semi_arid":        ([5, 5, 5, 4, 3, 1, 1, 1, 3, 4, 4, 5], 50),
    "desert":           ([2, 2, 2, 2, 1, 1, 0, 0, 1, 2, 2, 2], 35),
    "wet_mountain":     ([16, 14, 13, 11, 7, 3, 1, 2, 6, 10, 13, 16], 75),
    "oceanic":          ([12, 10, 11, 10, 11, 9, 8, 8, 9, 11, 12, 12], 80),
    "continental":      ([12, 10, 11, 11, 12, 11, 10, 10, 10, 11, 11, 12], 72),
    "continental_mild": ([10, 9, 10, 10, 11, 8, 7, 7, 8, 10, 10, 10], 74),
    "continental_cold": ([14, 11, 12, 12, 13, 12, 11, 10, 10, 12, 12, 13], 72),
    "dry_continental":  ([6, 6, 5, 7, 6, 3, 2, 2, 4, 7, 7, 7], 58),
    "euro_med":         ([7, 6, 6, 6, 5, 3, 2, 3, 5, 7, 7, 7], 65),
    "euro_med_wet":     ([14, 12, 11, 9, 7, 5, 3, 3, 6, 9, 11, 14], 72),
    "atlantic_med":     ([10, 9, 8, 9, 7, 3, 1, 1, 4, 8, 9, 11], 72),
    "gulf":             ([2, 2, 2, 1, 0, 0, 0, 0, 0, 0, 1, 2], 58),
}


def seasonal(month, v_low, v_high, low_month, high_month):
    """Smooth yearly curve through two known points (e.g. January and July values).
    Temperatures follow a cosine shape: coldest around January, warmest around late July."""
    peak = high_month + 0.3
    f = lambda m: math.cos(2 * math.pi * (m - peak) / 12)
    a = (v_high - v_low) / (f(high_month) - f(low_month))
    c = v_high - a * f(high_month)
    return round(c + a * f(month))


def condition(rain_days, temp_min, regime, month):
    if regime == "desert" and month in (3, 4):
        return "windy_sandy"            # spring sand winds in the Tunisian south
    if temp_min <= 0 and rain_days >= 8:
        return "snowy"
    if rain_days >= 10:
        return "rainy"
    if rain_days >= 5:
        return "partly_cloudy"
    return "sunny"


def make_weather():
    rows = []
    for city, (jmin, jmax, lmin, lmax, regime, sea) in CLIMATE.items():
        rain, hum = RAIN[regime]
        mean_rain = sum(rain) / 12
        for m in range(1, 13):
            tmin = seasonal(m, jmin, lmin, 1, 7)
            tmax = seasonal(m, jmax, lmax, 1, 7)
            rows.append({
                "city": city, "month": m, "temp_min": tmin, "temp_max": tmax,
                "condition": condition(rain[m - 1], tmin, regime, m),
                "rain_days": rain[m - 1],
                "humidity": max(20, min(95, round(hum + (rain[m - 1] - mean_rain) * 1.5))),
                "sea_temp": seasonal(m, sea[0], sea[1], 1, 8) if sea else "",
            })
    return rows


# =========================================================================== exchange rates
# Value of 1 unit of each currency in Tunisian dinars (mock values, plausible for 2026).
RATES_TO_TND = {"TND": 1.0, "EUR": 3.40, "USD": 3.05, "GBP": 3.95, "CHF": 3.55,
                "CAD": 2.20, "TRY": 0.075, "AED": 0.83, "SAR": 0.81}


# =========================================================================== hotels
# Price of a 3-star double room per night in TND (low season); other star levels are scaled from it.
CITY_PRICE_3STAR = {
    "Paris": 520, "London": 560, "Geneva": 600, "New York": 650, "Amsterdam": 500, "Rome": 420,
    "Milan": 450, "Barcelona": 420, "Madrid": 380, "Lisbon": 350, "Berlin": 370, "Frankfurt": 400,
    "Brussels": 380, "Marseille": 330, "Lyon": 320, "Malta": 300, "Istanbul": 230, "Dubai": 450,
    "Doha": 420, "Jeddah": 350, "Algiers": 220, "Casablanca": 240, "Montreal": 450,
}
TN_PRICE_3STAR = 160
STAR_FACTOR = {2: 0.65, 3: 1.0, 4: 1.6, 5: 2.8}
TN_RESORTS = {"Djerba", "Hammamet", "Sousse", "Monastir", "Mahdia", "Tabarka", "Tozeur", "Zarzis"}
TN_DESERT = {"Tozeur", "Douz", "Tataouine", "Matmata"}

HIGH_SEASON = {  # months when prices are higher
    "tn_coast": [6, 7, 8, 9], "tn_desert": [10, 11, 12, 1, 2, 3, 4], "europe": [6, 7, 8, 12],
    "gulf": [11, 12, 1, 2, 3], "other": [6, 7, 8],
}

NAME_WORDS_TN = ["Les Palmiers", "El Bahr", "Yasmine", "Corail", "Zitouna", "Le Phare", "Azur", "El Manar",
                 "Les Oliviers", "Jasmin", "Marhaba", "Sidi Salem", "La Perle", "Nour", "Les Mimosas", "Ulysse"]
NAME_PATTERNS_TN = ["Hôtel {w}", "{w} Beach Resort", "Résidence {w}", "Dar {w}", "{w} Palace", "Club {w}"]
NAME_WORDS_DESERT = ["Palmeraie", "Dunes d'Or", "Sahara Gate", "El Wahat", "Ras El Ain", "Les Sables"]
NAME_PATTERNS_DESERT = ["Oasis {w}", "{w} Desert Lodge", "Dar {w}", "Ksar {w}"]
NAME_WORDS_ABROAD = ["Central", "Park View", "Riverside", "Old Town", "Plaza", "Garden", "Harbour", "Lumière",
                     "Atlas", "Aurora", "Panorama", "Imperial", "Boulevard", "Station"]
NAME_PATTERNS_ABROAD = ["Hotel {w} {city}", "{city} {w} Hotel", "{w} Suites {city}", "Grand Hotel {w}"]


def season_of(city):
    p = PLACES[city]
    if p["region"] == "tunisia":
        return "tn_desert" if city in TN_DESERT else "tn_coast"
    if p["country"] in ("UAE", "Qatar", "Saudi Arabia"):
        return "gulf"
    if p["country"] in ("Algeria", "Morocco", "Turkey", "Canada", "USA"):
        return "other"
    return "europe"


def make_hotels(rng):
    rows, used_names = [], set()
    for city, p in PLACES.items():
        tunisian = p["region"] == "tunisia"
        coastal = CLIMATE[city][5] is not None
        n = 6 if city in TN_RESORTS else 3 if tunisian else 4
        if city in TN_DESERT:
            words, patterns = NAME_WORDS_DESERT, NAME_PATTERNS_DESERT
        elif tunisian:
            words, patterns = NAME_WORDS_TN, NAME_PATTERNS_TN
        else:
            words, patterns = NAME_WORDS_ABROAD, NAME_PATTERNS_ABROAD
        stars_pool = [3, 4, 4, 5, 5, 3] if city in TN_RESORTS else [2, 3, 4, 5]
        for i in range(n):
            while True:   # unique, fictional name
                name = rng.choice(patterns).format(w=rng.choice(words), city=city)
                if name not in used_names:
                    used_names.add(name)
                    break
            stars = stars_pool[i % len(stars_pool)]
            base = (TN_PRICE_3STAR if tunisian else CITY_PRICE_3STAR[city]) * STAR_FACTOR[stars]
            low = round(base * rng.uniform(0.85, 1.15) / 5) * 5
            high = round(low * (1.5 if tunisian else 1.3) / 5) * 5

            if tunisian and stars >= 4 and city in TN_RESORTS:
                board = rng.choice(["half_board", "all_inclusive", "all_inclusive"])
            elif tunisian:
                board = rng.choice(["bed_and_breakfast", "half_board"])
            else:
                board = rng.choice(["room_only", "bed_and_breakfast", "bed_and_breakfast"])

            fac = ["wifi"]
            if stars >= 3: fac.append("restaurant")
            if stars >= 4 or (coastal and rng.random() < 0.5): fac.append("pool")
            if stars >= 4: fac += ["gym", "room_service", "air_conditioning"]
            elif rng.random() < 0.6: fac.append("air_conditioning")
            if stars == 5 and rng.random() < 0.8: fac.append("spa")
            if coastal and city in TN_RESORTS | {"Malta", "Dubai", "Barcelona"} and rng.random() < 0.6:
                fac.append("private_beach")
            if city in TN_RESORTS and stars >= 4 and rng.random() < 0.7: fac.append("kids_club")
            if tunisian and stars >= 4 and rng.random() < 0.6: fac.append("hammam")
            if rng.random() < 0.7: fac.append("parking")
            if rng.random() < 0.2: fac.append("pets_allowed")
            if rng.random() < 0.3 + 0.1 * stars: fac.append("wheelchair_access")
            if rng.random() < 0.4: fac.append("airport_shuttle")
            if "restaurant" in fac and (tunisian or season_of(city) == "gulf" or city in ("Istanbul", "Algiers", "Casablanca")
                                        or rng.random() < 0.15):
                fac.append("halal_food")
            if board != "room_only": fac.append("breakfast_included")

            rows.append({
                "city": city, "name": name, "stars": stars,
                "price_low_season_tnd": low, "price_high_season_tnd": high,
                "high_season_months": ",".join(map(str, HIGH_SEASON[season_of(city)])),
                "board": board, "facilities": ";".join(fac),
                "rating": round(min(9.6, rng.uniform(5.8, 7.2) + 0.45 * (stars - 2)), 1),
            })
    return rows


# =========================================================================== flights
AIRPORT_NAMES = {
    "TUN": "Tunis-Carthage", "DJE": "Djerba-Zarzis", "MIR": "Monastir Habib Bourguiba",
    "NBE": "Enfidha-Hammamet", "SFA": "Sfax-Thyna", "TOE": "Tozeur-Nefta", "TBJ": "Tabarka-Ain Draham",
    "GAE": "Gabes-Matmata", "CDG": "Paris-Charles de Gaulle", "MRS": "Marseille-Provence", "LYS": "Lyon-Saint Exupéry",
    "FCO": "Rome-Fiumicino", "MXP": "Milan-Malpensa", "IST": "Istanbul", "DXB": "Dubai International",
    "LHR": "London-Heathrow", "MAD": "Madrid-Barajas", "BCN": "Barcelona-El Prat", "BER": "Berlin-Brandenburg",
    "FRA": "Frankfurt", "BRU": "Brussels", "GVA": "Geneva", "AMS": "Amsterdam-Schiphol", "LIS": "Lisbon",
    "MLA": "Malta International", "ALG": "Algiers-Houari Boumediene", "CMN": "Casablanca-Mohammed V",
    "JED": "Jeddah-King Abdulaziz", "DOH": "Doha-Hamad", "YUL": "Montreal-Trudeau", "JFK": "New York-JFK",
}

# airline -> (IATA code, price factor, cabin bag kg, checked bag kg (0 = paid only), extra bag price TND)
AIRLINES = {
    "Tunisair": ("TU", 1.00, 8, 23, 150), "Tunisair Express": ("UG", 1.00, 8, 20, 80),
    "Nouvelair": ("BJ", 0.85, 8, 23, 140), "Transavia": ("TO", 0.75, 10, 0, 120),
    "Air France": ("AF", 1.25, 12, 23, 220), "Lufthansa": ("LH", 1.25, 8, 23, 220),
    "Turkish Airlines": ("TK", 1.05, 8, 30, 200), "Emirates": ("EK", 1.30, 7, 30, 250),
    "Qatar Airways": ("QR", 1.20, 7, 30, 250), "Saudia": ("SV", 1.05, 7, 46, 200),
    "ITA Airways": ("AZ", 1.15, 8, 23, 200), "Air Algérie": ("AH", 0.95, 10, 30, 150),
    "Royal Air Maroc": ("AT", 1.00, 10, 23, 180),
}

# Flight time from Tunis in minutes (other Tunisian airports add a small offset)
DURATION_FROM_TUN = {
    "CDG": 145, "MRS": 100, "LYS": 110, "FCO": 75, "MXP": 105, "IST": 165, "DXB": 360, "LHR": 170,
    "MAD": 125, "BCN": 105, "BER": 165, "FRA": 140, "BRU": 155, "GVA": 115, "AMS": 170, "LIS": 175,
    "MLA": 55, "ALG": 80, "CMN": 165, "JED": 260, "DOH": 285, "YUL": 520,
}
OFFSET = {"TUN": 0, "DJE": 25, "MIR": 5, "NBE": 5, "SFA": 15, "TOE": 20}
DOMESTIC = {("TUN", "DJE"): 65, ("TUN", "TOE"): 65, ("TUN", "SFA"): 55, ("TUN", "TBJ"): 45, ("TUN", "GAE"): 60}

# (origin, destination, airlines, flights per week, season)  -- the return direction is added automatically
ROUTES = [
    ("TUN", "CDG", ["Tunisair", "Air France", "Transavia", "Nouvelair"], 7, "all_year"),
    ("TUN", "MRS", ["Tunisair", "Nouvelair"], 4, "all_year"),
    ("TUN", "LYS", ["Tunisair", "Transavia"], 4, "all_year"),
    ("TUN", "FCO", ["Tunisair", "ITA Airways"], 5, "all_year"),
    ("TUN", "MXP", ["Tunisair"], 4, "all_year"),
    ("TUN", "IST", ["Turkish Airlines", "Tunisair"], 7, "all_year"),
    ("TUN", "DXB", ["Emirates"], 7, "all_year"),
    ("TUN", "LHR", ["Tunisair"], 3, "all_year"),
    ("TUN", "MAD", ["Tunisair"], 3, "all_year"),
    ("TUN", "BCN", ["Tunisair"], 3, "all_year"),
    ("TUN", "BER", ["Tunisair"], 2, "all_year"),
    ("TUN", "FRA", ["Lufthansa", "Tunisair"], 5, "all_year"),
    ("TUN", "BRU", ["Tunisair"], 4, "all_year"),
    ("TUN", "GVA", ["Tunisair"], 3, "all_year"),
    ("TUN", "AMS", ["Tunisair"], 2, "all_year"),
    ("TUN", "LIS", ["Tunisair"], 2, "all_year"),
    ("TUN", "MLA", ["Tunisair"], 2, "all_year"),
    ("TUN", "ALG", ["Tunisair", "Air Algérie"], 5, "all_year"),
    ("TUN", "CMN", ["Royal Air Maroc", "Tunisair"], 4, "all_year"),
    ("TUN", "JED", ["Tunisair", "Saudia"], 4, "all_year"),
    ("TUN", "DOH", ["Qatar Airways"], 7, "all_year"),
    ("TUN", "YUL", ["Tunisair"], 2, "all_year"),
    ("TUN", "DJE", ["Tunisair Express"], 7, "all_year"),
    ("TUN", "TOE", ["Tunisair Express"], 3, "all_year"),
    ("TUN", "SFA", ["Tunisair Express"], 3, "all_year"),
    ("TUN", "TBJ", ["Tunisair Express"], 2, "summer"),
    ("TUN", "GAE", ["Tunisair Express"], 2, "all_year"),
    ("DJE", "CDG", ["Transavia", "Nouvelair", "Tunisair"], 4, "all_year"),
    ("DJE", "LYS", ["Transavia", "Nouvelair"], 3, "all_year"),
    ("DJE", "MRS", ["Nouvelair", "Tunisair"], 3, "all_year"),
    ("DJE", "IST", ["Turkish Airlines"], 3, "all_year"),
    ("DJE", "FRA", ["Tunisair"], 2, "summer"),
    ("DJE", "BRU", ["Nouvelair"], 2, "summer"),
    ("DJE", "GVA", ["Nouvelair"], 2, "summer"),
    ("MIR", "CDG", ["Nouvelair", "Transavia"], 4, "all_year"),
    ("MIR", "LYS", ["Nouvelair"], 3, "summer"),
    ("MIR", "MRS", ["Nouvelair"], 2, "summer"),
    ("MIR", "BRU", ["Nouvelair"], 2, "summer"),
    ("MIR", "FRA", ["Nouvelair"], 2, "summer"),
    ("NBE", "CDG", ["Nouvelair", "Transavia"], 3, "summer"),
    ("NBE", "LHR", ["Nouvelair"], 2, "summer"),
    ("NBE", "BRU", ["Nouvelair"], 2, "summer"),
    ("NBE", "FRA", ["Nouvelair"], 2, "summer"),
    ("NBE", "AMS", ["Nouvelair"], 2, "summer"),
    ("SFA", "CDG", ["Tunisair", "Nouvelair"], 3, "all_year"),
    ("SFA", "LYS", ["Tunisair"], 2, "all_year"),
    ("SFA", "MRS", ["Tunisair"], 2, "all_year"),
    ("TOE", "CDG", ["Tunisair"], 1, "winter"),
]
# One-stop connections to New York (no direct flight from Tunisia)
CONNECTIONS = [("Turkish Airlines", "IST", 780), ("Qatar Airways", "DOH", 900), ("Air France", "CDG", 690)]


def price_for(duration, airline, domestic):
    base = 70 + 1.2 * duration if domestic else 90 + 3.2 * duration
    return round(base * AIRLINES[airline][1] / 5) * 5


def make_flights(rng):
    rows, used_numbers = [], set()

    def flight_no(airline):
        code = AIRLINES[airline][0]
        while True:
            n = f"{code}{rng.randint(100, 999 if code != 'UG' else 699)}"
            if n not in used_numbers:
                used_numbers.add(n)
                return n

    def add(orig, dest, airline, per_week, season, duration, stops=0, via=""):
        domestic = PLACES_BY_AIRPORT.get(dest, "abroad") == "tunisia" and PLACES_BY_AIRPORT.get(orig, "abroad") == "tunisia"
        days = sorted(rng.sample(range(7), per_week))       # 0 = Monday ... 6 = Sunday
        hour, minute = rng.randint(6, 21), rng.choice(range(0, 60, 5))
        _, _, cabin, checked, extra = AIRLINES[airline]
        rows.append({
            "flight_no": flight_no(airline), "airline": airline,
            "origin": orig, "origin_name": AIRPORT_NAMES[orig],
            "destination": dest, "destination_name": AIRPORT_NAMES[dest],
            "days_of_week": ",".join(map(str, days)), "departure_time": f"{hour:02d}:{minute:02d}",
            "duration_min": duration, "stops": stops, "via": via, "season": season,
            "base_price_tnd": price_for(duration, airline, domestic),
            "cabin_bag_kg": cabin, "checked_bag_kg": checked, "extra_bag_price_tnd": extra,
        })

    for orig, dest, airlines, per_week, season in ROUTES:
        if (orig, dest) in DOMESTIC:
            duration = DOMESTIC[(orig, dest)]
        else:
            duration = DURATION_FROM_TUN[dest] + OFFSET[orig]
        for i, airline in enumerate(airlines):
            # the first airline listed on a route is the main one; the others fly less often
            n = per_week if i == 0 else max(1, per_week - rng.randint(1, 3))
            add(orig, dest, airline, n, season, duration)
            add(dest, orig, airline, n, season, duration + rng.choice([0, 5, 10, 15]))  # return flight
            if (orig, dest) == ("TUN", "DJE"):   # busiest domestic route: a second daily rotation
                add(orig, dest, airline, n, season, duration)
                add(dest, orig, airline, n, season, duration)

    for airline, hub, duration in CONNECTIONS:
        add("TUN", "JFK", airline, 7 if airline != "Air France" else 4, "all_year", duration, 1, hub)
        add("JFK", "TUN", airline, 7 if airline != "Air France" else 4, "all_year", duration + 30, 1, hub)
    return rows


PLACES_BY_AIRPORT = {"TUN": "tunisia", "DJE": "tunisia", "MIR": "tunisia", "NBE": "tunisia", "SFA": "tunisia",
                     "TOE": "tunisia", "TBJ": "tunisia", "GAE": "tunisia"}


# =========================================================================== main

def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def main():
    rng = random.Random(SEED)
    OUT.mkdir(parents=True, exist_ok=True)

    weather = make_weather()
    write_csv(OUT / "weather.csv", weather)

    with open(OUT / "exchange_rates.json", "w", encoding="utf-8") as f:
        json.dump({"_note": "MOCK rates. Value of 1 unit of each currency in TND. Convert A->B via TND.",
                   "base": "TND", "as_of": "2026-11-01", "rates_to_tnd": RATES_TO_TND}, f, indent=2)

    hotels = make_hotels(rng)
    write_csv(OUT / "hotels.csv", hotels)

    flights = make_flights(rng)
    write_csv(OUT / "flights.csv", flights)

    # sanity: every place we can talk about has weather and hotels, and every place's airport is served
    served = {r["origin"] for r in flights} | {r["destination"] for r in flights}
    missing_airports = sorted({p["airport"] for p in PLACES.values()} - served)
    print(f"weather.csv: {len(weather)} rows ({len(CLIMATE)} cities x 12 months)")
    print(f"hotels.csv:  {len(hotels)} hotels")
    print(f"flights.csv: {len(flights)} timetable rows")
    print("exchange_rates.json: ", ", ".join(RATES_TO_TND))
    print("Places without weather:", sorted(set(PLACES) - set(CLIMATE)) or "none")
    print("Airports from places.json with no flights:", missing_airports or "none")


if __name__ == "__main__":
    main()
