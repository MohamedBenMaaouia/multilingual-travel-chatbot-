"""Official label names shared by every module. Import from here instead of retyping strings."""

LANGS = ["en", "fr", "tn_latin"]

# category -> its 3 subcategories
TAXONOMY = {
    "flights": ["flight_search", "flight_price", "baggage"],
    "accommodation": ["hotel_search", "hotel_price", "hotel_facilities"],
    "local_transport": ["car_rental", "transfer_taxi", "public_transport"],
    "destination_info": ["weather", "activities", "visa_documents"],
    "budget_money": ["exchange_rate", "trip_cost", "payment_methods"],
}

# subcategory -> category (the classifier predicts the subcategory; the category is looked up here)
SUBCAT_TO_CAT = {sub: cat for cat, subs in TAXONOMY.items() for sub in subs}
SUBCATEGORIES = list(SUBCAT_TO_CAT)

ENTITY_LABELS = ["LOC", "DATE"]
