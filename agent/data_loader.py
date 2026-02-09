"""
KDD Travel Planning - Data Loader Module
=========================================
"""

import os
import ast
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Any


CUR_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(os.path.dirname(CUR_DIR), "api")
DATA_DIR = os.path.join(API_DIR, "data")

FLIGHT_CSV_PATH = os.path.join(DATA_DIR, "flight_data", "flights.csv")
HOTEL_DATA_DIR = os.path.join(DATA_DIR, "hotel_data")
ATTRACTION_DATA_DIR = os.path.join(DATA_DIR, "attraction_data")
CAR_DATA_DIR = os.path.join(DATA_DIR, "car_data")


@dataclass
class QueryMeta:
    query: str
    level: str  # easy/medium/hard
    person_num: int = 1
    budget: float = 0.0
    days: int = 3
    rooms_count: int = 1
    implicit_keywords: list[str] = field(default_factory=list)
    hard_constraints: dict = field(default_factory=dict)
    req_flight: dict = field(default_factory=dict)
    req_hotel: dict = field(default_factory=dict)
    req_car: dict = field(default_factory=dict)
    req_attraction: dict = field(default_factory=dict)
    cities_count: int = 1
    is_ordered: bool = False
    impossible: bool = False
    
    @classmethod
    def from_csv_row(cls, row: pd.Series) -> "QueryMeta":
        def safe_eval(val, default=None):
            if pd.isna(val) or val == "" or val == "nan":
                return default
            try:
                return ast.literal_eval(str(val))
            except:
                return default
        
        hard_constraints = safe_eval(row.get("hard_constraints"), {})
        
        implicit_kw_raw = safe_eval(row.get("implicit_keywords"), [])
        implicit_keywords = implicit_kw_raw if isinstance(implicit_kw_raw, list) else []
        
        req_flight = safe_eval(row.get("req_flight"), {})
        
        req_hotel = safe_eval(row.get("req_hotel"), {})
        
        req_car = safe_eval(row.get("req_car"), {})
        
        req_attraction = safe_eval(row.get("req_attraction"), {})
        
        cities_count_raw = safe_eval(row.get("cities_count"), {"cities_count": 1})
        cities_count = cities_count_raw.get("cities_count", 1) if isinstance(cities_count_raw, dict) else 1
        
        impossible = str(row.get("impossible", "")).lower() in ("true", "1", "yes", "1.0")
        
        def to_int(val, default=0):
            try:
                return int(val)
            except (ValueError, TypeError):
                return default
        
        def to_float(val, default=0.0):
            try:
                return float(val)
            except (ValueError, TypeError):
                return default
        
        query_text = str(row.get("query", "")).lower()
        is_ordered = any(keyword in query_text for keyword in [
            "first", "then", "after", "before", "next", "finally",
            "day 1", "day 2", "day 3", "day1", "day2", "day3",
            "先", "然后", "接着", "之后", "最后"
        ])
        
        if cities_count > 1 and is_ordered:
            is_ordered = True
        else:
            is_ordered = False
        
        return cls(
            query=str(row.get("query", "")),
            level=str(row.get("level", "easy")),
            person_num=to_int(hard_constraints.get("person_num", 1), 1),
            budget=to_float(hard_constraints.get("budget", 0), 0),
            days=to_int(hard_constraints.get("days", 3), 3),
            rooms_count=to_int(req_hotel.get("rooms_count", 1), 1) if req_hotel else 1,
            implicit_keywords=implicit_keywords,
            hard_constraints=hard_constraints,
            req_flight=req_flight,
            req_hotel=req_hotel,
            req_car=req_car,
            req_attraction=req_attraction,
            cities_count=cities_count,
            is_ordered=is_ordered,
            impossible=impossible,
        )


def load_queries(csv_path: str) -> list[QueryMeta]:
    df = pd.read_csv(csv_path)
    return [QueryMeta.from_csv_row(row) for _, row in df.iterrows()]


class SandboxDB:
    
    def __init__(self):
        self._flights_df: Optional[pd.DataFrame] = None
        self._hotels_cache: dict[str, pd.DataFrame] = {}
        self._attractions_cache: dict[str, pd.DataFrame] = {}
        self._cars_cache: dict[str, pd.DataFrame] = {}
    
    @property
    def flights(self) -> pd.DataFrame:
        if self._flights_df is None:
            if os.path.exists(FLIGHT_CSV_PATH):
                self._flights_df = pd.read_csv(FLIGHT_CSV_PATH)
            else:
                self._flights_df = pd.DataFrame()
        return self._flights_df
    
    def get_hotels(self, city: str) -> pd.DataFrame:
        if city not in self._hotels_cache:
            csv_path = os.path.join(HOTEL_DATA_DIR, f"{city}_hotel.csv")
            if os.path.exists(csv_path):
                self._hotels_cache[city] = pd.read_csv(csv_path)
            else:
                self._hotels_cache[city] = pd.DataFrame()
        return self._hotels_cache[city]
    
    def get_attractions(self, city: str) -> pd.DataFrame:
        if city not in self._attractions_cache:
            csv_path = os.path.join(ATTRACTION_DATA_DIR, f"{city}_attraction.csv")
            if os.path.exists(csv_path):
                self._attractions_cache[city] = pd.read_csv(csv_path)
            else:
                self._attractions_cache[city] = pd.DataFrame()
        return self._attractions_cache[city]
    
    def get_cars(self, city: str) -> pd.DataFrame:
        if city not in self._cars_cache:
            csv_path = os.path.join(CAR_DATA_DIR, f"{city}_rental_cars.csv")
            if os.path.exists(csv_path):
                self._cars_cache[city] = pd.read_csv(csv_path)
            else:
                self._cars_cache[city] = pd.DataFrame()
        return self._cars_cache[city]
    
    def verify_flight(self, flight_number: str, departure_city: str = None, 
                      arrival_city: str = None, price: float = None) -> bool:
        df = self.flights
        if df.empty:
            return False
        
        mask = df["flight_number"] == flight_number
        if departure_city:
            mask &= df["departure_city"] == departure_city
        if arrival_city:
            mask &= df["arrival_city"] == arrival_city
        
        matches = df[mask]
        if matches.empty:
            return False
        
        if price is not None:
            return any(abs(matches["price"] - price) < 0.01)
        return True
    
    def verify_hotel(self, name: str, city: str, price: float = None) -> bool:
        df = self.get_hotels(city)
        if df.empty:
            return False
        
        matches = df[df["name"].str.lower() == name.lower()]
        if matches.empty:
            return False
        
        if price is not None:
            return any(abs(matches["price"] - price) < 0.01)
        return True
    
    def verify_attraction(self, name: str, city: str) -> bool:
        df = self.get_attractions(city)
        if df.empty:
            return False
        
        return any(df["attraction_name"].str.lower() == name.lower())
    
    def get_open_hours(self, attraction_name: str, city: str) -> Optional[str]:
        df = self.get_attractions(city)
        if df.empty:
            return None
        
        matches = df[df["attraction_name"].str.lower() == attraction_name.lower()]
        if matches.empty:
            return None
        
        return str(matches.iloc[0].get("open_hours", ""))
    
    def get_coordinates(self, entity_name: str, entity_type: str, city: str) -> Optional[tuple[float, float]]:
        if entity_type == "hotel":
            df = self.get_hotels(city)
            name_col = "name"
        elif entity_type == "attraction":
            df = self.get_attractions(city)
            name_col = "attraction_name"
        elif entity_type == "flight":
            df = self.flights
            matches = df[df["flight_number"] == entity_name]
            if matches.empty:
                return None
            row = matches.iloc[0]
            return (row.get("arrival_airport_latitude"), row.get("arrival_airport_longitude"))
        else:
            return None
        
        if df.empty:
            return None
        
        matches = df[df[name_col].str.lower() == entity_name.lower()]
        if matches.empty:
            return None
        
        row = matches.iloc[0]
        lat = row.get("latitude")
        lon = row.get("longitude")
        if pd.isna(lat) or pd.isna(lon):
            return None
        return (float(lat), float(lon))
    
    def get_flight_price(self, flight_number: str) -> Optional[float]:
        df = self.flights
        if df.empty:
            return None
        
        matches = df[df["flight_number"] == flight_number]
        if matches.empty:
            return None
        
        price = matches.iloc[0].get("price")
        return float(price) if pd.notna(price) else None
    
    def get_hotel_price(self, name: str, city: str = None) -> Optional[float]:
        if city:
            df = self.get_hotels(city)
            if not df.empty:
                matches = df[df["name"].str.lower() == name.lower()]
                if not matches.empty:
                    price = matches.iloc[0].get("price")
                    return float(price) if pd.notna(price) else None
        
        for csv_file in os.listdir(HOTEL_DATA_DIR):
            if csv_file.endswith("_hotel.csv"):
                city_name = csv_file.replace("_hotel.csv", "")
                df = self.get_hotels(city_name)
                if not df.empty:
                    matches = df[df["name"].str.lower() == name.lower()]
                    if not matches.empty:
                        price = matches.iloc[0].get("price")
                        return float(price) if pd.notna(price) else None
        
        return None
    
    def get_attraction_price(self, name: str, city: str = None) -> Optional[float]:
        if city:
            df = self.get_attractions(city)
            if not df.empty:
                matches = df[df["attraction_name"].str.lower() == name.lower()]
                if not matches.empty:
                    price = matches.iloc[0].get("ticket_price")
                    return float(price) if pd.notna(price) else None
        
        for csv_file in os.listdir(ATTRACTION_DATA_DIR):
            if csv_file.endswith("_attraction.csv"):
                city_name = csv_file.replace("_attraction.csv", "")
                df = self.get_attractions(city_name)
                if not df.empty:
                    matches = df[df["attraction_name"].str.lower() == name.lower()]
                    if not matches.empty:
                        price = matches.iloc[0].get("ticket_price")
                        return float(price) if pd.notna(price) else None
        
        return None
    
    def get_car_price(self, car_type: str = None, city: str = None) -> Optional[float]:
        if city:
            df = self.get_cars(city)
            if not df.empty:
                if car_type:
                    matches = df[df["car_type"].str.lower() == car_type.lower()]
                else:
                    matches = df
                if not matches.empty:
                    price = matches.iloc[0].get("price_per_day")
                    return float(price) if pd.notna(price) else None
        
        for csv_file in os.listdir(CAR_DATA_DIR):
            if csv_file.endswith("_rental_cars.csv"):
                city_name = csv_file.replace("_rental_cars.csv", "")
                df = self.get_cars(city_name)
                if not df.empty:
                    if car_type:
                        matches = df[df["car_type"].str.lower() == car_type.lower()]
                    else:
                        matches = df
                    if not matches.empty:
                        price = matches.iloc[0].get("price_per_day")
                        return float(price) if pd.notna(price) else None
        
        return None


_sandbox_db: Optional[SandboxDB] = None

def get_sandbox_db() -> SandboxDB:
    global _sandbox_db
    if _sandbox_db is None:
        _sandbox_db = SandboxDB()
    return _sandbox_db


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        queries = load_queries(csv_path)
        print(f"Loaded {len(queries)} queries from {csv_path}")
        if queries:
            q = queries[0]
            print(f"First query: {q.query[:80]}...")
            print(f"  Level: {q.level}, Budget: {q.budget}, Days: {q.days}")
            print(f"  Implicit keywords: {q.implicit_keywords}")
    else:
        db = get_sandbox_db()
        print(f"Flights loaded: {len(db.flights)} rows")
