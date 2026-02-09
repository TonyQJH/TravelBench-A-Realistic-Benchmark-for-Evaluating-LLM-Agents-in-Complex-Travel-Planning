
import os
import sys
import json
import math
import argparse
import re
from dataclasses import dataclass, field, asdict
from typing import Optional, Any
from datetime import datetime

import pandas as pd

from data_loader import QueryMeta, load_queries, SandboxDB, get_sandbox_db
from implicit_scoring import score_d1_implicit_v2, get_implicit_scorer, get_embedding_model


@dataclass
class ScoreResult:
    query_index: int
    
    # D0: Explicit
    d0_keyword: Optional[float] = None
    d0_source: Optional[float] = None
    
    # D1: Implicit
    d1_implicit: Optional[float] = None
    
    # D2: City Dimension (2 columns)
    d2_unord_single: Optional[float] = None
    d2_unord_multi: Optional[float] = None
    
    # D3: Budget Trade-off
    d3_budget: Optional[float] = None
    
    # D4: Impossible Handling
    d4_impossible: Optional[float] = None
    
    # D5: Retry Robustness
    d5_retry: Optional[float] = None
    
    # CCR
    b2_opening_hours: Optional[float] = None
    b3_spatiotemporal: Optional[float] = None
    
    total_cost: Optional[float] = None
    attraction_count: int = 0
    is_feasible: bool = False


def parse_time(time_str: str) -> Optional[int]:
    if not time_str or time_str == "-":
        return None
    try:
        parts = time_str.split(":")
        return int(parts[0]) * 60 + int(parts[1])
    except:
        return None


def is_time_in_range(visit_time: str, open_hours: str) -> bool:
    if not open_hours or open_hours == "-":
        return True
    
    visit_min = parse_time(visit_time)
    if visit_min is None:
        return True
    
    try:
        parts = open_hours.split("-")
        if len(parts) != 2:
            return True
        open_min = parse_time(parts[0].strip())
        close_min = parse_time(parts[1].strip())
        if open_min is None or close_min is None:
            return True
        return open_min <= visit_min <= close_min
    except:
        return True


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371
    
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lat = math.radians(lat2 - lat1)
    delta_lon = math.radians(lon2 - lon1)
    
    a = math.sin(delta_lat/2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    return R * c


def get_min_travel_time(distance_km: float) -> int:
    if distance_km <= 30:
        return 30
    elif distance_km <= 300:
        return int(distance_km / 60 + 30)
    else:
        return 60


class TravelPlanScorer:
    
    def __init__(self, db: SandboxDB, config: dict = None):
        self.db = db
        self.config = config or {}
        
        self.beta = self.config.get("beta", 4.0)
        self.beta = self.config.get("beta", 4.0)
    
    def extract_plan_data(self, plan_output: dict) -> dict:
        is_feasible = plan_output.get("is_feasible", False)
        
        plan = plan_output.get("plan") or plan_output.get("Plan", {})
        
        flights = []
        for day_key, day_data in plan.items():
            if isinstance(day_data, dict):
                day_flights = day_data.get("flights") or day_data.get("Flight", [])
                if isinstance(day_flights, list):
                    for f in day_flights:
                        if isinstance(f, dict) and f.get("flight_number"):
                            flights.append(f)
                        elif isinstance(f, str) and f != "-":
                            pass
        
        attractions = []
        for day_key, day_data in plan.items():
            if isinstance(day_data, dict):
                day_attrs = day_data.get("attractions") or day_data.get("Attraction", [])
                if isinstance(day_attrs, list):
                    for a in day_attrs:
                        if isinstance(a, dict):
                            name = a.get("name") or a.get("attraction_name")
                            if name and name != "-":
                                attractions.append({**a, "name": name})
        
        hotels = []
        hotel_keys_seen = set()
        for day_key, day_data in plan.items():
            if isinstance(day_data, dict):
                hotel = day_data.get("hotel") or day_data.get("Hotel")
                if isinstance(hotel, dict):
                    name = hotel.get("name")
                    city = hotel.get("city", "")
                    if name and name != "-":
                        hotel_key = (name.lower(), city.lower())
                        if hotel_key not in hotel_keys_seen:
                            hotel_keys_seen.add(hotel_key)
                            hotels.append(hotel)
        
        cars = []
        car_keys_seen = set()
        for day_key, day_data in plan.items():
            if isinstance(day_data, dict):
                car = day_data.get("car") or day_data.get("Car")
                if isinstance(car, dict):
                    car_type = car.get("type") or car.get("car_type")
                    city = car.get("city", "")
                    if car_type and car_type != "-":
                        car_key = (car_type.lower(), city.lower())
                        if car_key not in car_keys_seen:
                            car_keys_seen.add(car_key)
                            cars.append(car)
        
        cities = []
        for day_key in sorted(plan.keys()):
            day_data = plan.get(day_key, {})
            if isinstance(day_data, dict):
                current = day_data.get("current_city") or day_data.get("Current City", "")
                if current:
                    if "to" in current.lower():
                        parts = re.split(r'\s+to\s+', current, flags=re.IGNORECASE)
                        if len(parts) == 2:
                            if parts[0].lower().startswith("from "):
                                parts[0] = parts[0][5:]
                            cities.extend([p.strip() for p in parts])
                        else:
                            cities.append(current)
                    else:
                        cities.append(current)
        
        unique_cities = []
        for c in cities:
            if c and c not in unique_cities:
                unique_cities.append(c)
        
        return {
            "is_feasible": is_feasible,
            "flights": flights,
            "attractions": attractions,
            "hotels": hotels,
            "cars": cars,
            "cities": unique_cities,
            "plan": plan
        }
    
    def score_d0_keyword(self, plan_data: dict, meta: QueryMeta) -> float:
        checks = []
        matched = 0
        
        if meta.req_flight:
            dep_city = meta.req_flight.get("departure_city", "")
            if dep_city:
                checks.append(("departure_city", dep_city))
                if any(f.get("departure_city", "").lower() == dep_city.lower() 
                       for f in plan_data["flights"]):
                    matched += 1
            
            arr_cities = meta.req_flight.get("arrival_city", [])
            if isinstance(arr_cities, str):
                arr_cities = [arr_cities]
            elif not isinstance(arr_cities, list):
                arr_cities = []
            
            for arr_city in arr_cities:
                if arr_city:
                    checks.append(("arrival_city", arr_city))
                    has_flight = any(f.get("arrival_city", "").lower() == arr_city.lower() 
                                    for f in plan_data["flights"])
                    in_cities = any(c.lower() == arr_city.lower() for c in plan_data["cities"])
                    if has_flight or in_cities:
                        matched += 1
        
        if meta.req_hotel:
            req_name = meta.req_hotel.get("name") or meta.req_hotel.get("hotel_name")
            if req_name and req_name != "-":
                checks.append(("hotel_name", req_name))
                if any(h.get("name", "").lower() == req_name.lower() for h in plan_data["hotels"]):
                    matched += 1
        
        if meta.req_car:
            req_type = meta.req_car.get("car_type") or meta.req_car.get("type")
            if req_type and req_type != "-":
                checks.append(("car_type", req_type))
                if any((c.get("type") or c.get("car_type") or "").lower() == req_type.lower() 
                       for c in plan_data["cars"]):
                    matched += 1
        
        if meta.req_attraction:
            req_attrs = []
            raw_attr = meta.req_attraction.get("attraction_name") or meta.req_attraction.get("name")
            
            if isinstance(raw_attr, list):
                req_attrs = raw_attr
            elif isinstance(raw_attr, str) and raw_attr != "-":
                req_attrs = [raw_attr]
                
            for attr_name in req_attrs:
                if attr_name:
                    checks.append(("attraction_name", attr_name))
                    if any(a.get("name", "").lower() == attr_name.lower() for a in plan_data["attractions"]):
                        matched += 1
        
        if meta.implicit_keywords and any(k.lower() in ["road trip", "roadtrip"] for k in meta.implicit_keywords):
            checks.append(("road_trip", "daily_car"))
            
            days_with_car = 0
            plan_dict = plan_data.get("plan", {})
            
            for day_content in plan_dict.values():
                if isinstance(day_content, dict):
                    car = day_content.get("car") or day_content.get("Car")
                    if car and car != "-":
                         days_with_car += 1
            
            target_days = meta.days
            if days_with_car >= target_days:
                matched += 1

        return matched / len(checks) if checks else None
    
    def score_d0_source(self, plan_data: dict, meta: QueryMeta) -> float:
        facts_checked = 0
        facts_matched = 0
        
        for flight in plan_data["flights"]:
            fn = flight.get("flight_number")
            if fn:
                facts_checked += 1
                if self.db.verify_flight(fn, 
                    departure_city=flight.get("departure_city"),
                    arrival_city=flight.get("arrival_city")):
                    facts_matched += 1
        
        for hotel in plan_data["hotels"]:
            name = hotel.get("name")
            city = hotel.get("city")
            if name and city:
                facts_checked += 1
                if self.db.verify_hotel(name, city):
                    facts_matched += 1
        
        for attr in plan_data["attractions"]:
            name = attr.get("name")
            city = attr.get("city")
            if name and city:
                facts_checked += 1
                if self.db.verify_attraction(name, city):
                    facts_matched += 1
        
        if facts_checked > 0:
            return facts_matched / facts_checked
        else:
            if plan_data["is_feasible"]:
                return 0.0
            else:
                return None
    
    def score_d1_implicit(self, plan_data: dict, meta: QueryMeta) -> float:
        implicit_kw = meta.implicit_keywords
        if not implicit_kw:
            return None
        
        return score_d1_implicit_v2(plan_data, implicit_kw)
    
    def score_d2_city(self, plan_data: dict, meta: QueryMeta) -> dict[str, float]:
        expected_cities = []
        if meta.req_flight:
            arr = meta.req_flight.get("arrival_city", [])
            if isinstance(arr, list):
                expected_cities = arr
            elif arr:
                expected_cities = [arr]
        
        actual_cities = plan_data["cities"]
        
        if not expected_cities:
            score = 1.0 if actual_cities else 0.0
        else:
            if meta.is_ordered:
                score = self._compute_city_order_score(expected_cities, actual_cities)
            else:
                actual_lower = [c.lower() for c in actual_cities]
                matched = sum(1 for c in expected_cities if c.lower() in actual_lower)
                score = matched / len(expected_cities)
        
        # is_ordered = meta.is_ordered  # Ignored as per user request (confirmed no ordered queries)
        num_cities = meta.cities_count
        
        result = {}
        if num_cities == 1:
            result["d2_unord_single"] = score
        else:
            result["d2_unord_multi"] = score
        
        return result
    
    def _compute_city_order_score(self, expected: list[str], actual: list[str]) -> float:
        if not expected:
            return 1.0
        if not actual:
            return 0.0
        
        m, n = len(expected), len(actual)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if expected[i-1] == actual[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        lcs_length = dp[m][n]
        
        return lcs_length / len(expected)
    
    def compute_total_cost(self, plan_data: dict, meta: QueryMeta) -> float:
        total = 0.0
        
        for flight in plan_data["flights"]:
            fn = flight.get("flight_number")
            if fn:
                db_price = self.db.get_flight_price(fn)
                if db_price is not None:
                    total += db_price * meta.person_num
                else:
                    price = flight.get("price", 0)
                    if isinstance(price, (int, float)):
                        total += price * meta.person_num
        
        nights = meta.days - 1
        if nights < 1:
            nights = 1
        hotel_keys_seen = set()
        for hotel in plan_data["hotels"]:
            name = hotel.get("name")
            city = hotel.get("city")
            if name:
                hotel_key = (name, city) if city else (name, "")
                if hotel_key not in hotel_keys_seen:
                    hotel_keys_seen.add(hotel_key)
                    db_price = self.db.get_hotel_price(name, city)
                    if db_price is not None:
                        total += db_price * meta.rooms_count * nights
                    else:
                        price = hotel.get("price_per_night") or hotel.get("price", 0)
                        if isinstance(price, (int, float)):
                            total += price * meta.rooms_count * nights
        
        for attr in plan_data["attractions"]:
            name = attr.get("name")
            city = attr.get("city")
            if name:
                db_price = self.db.get_attraction_price(name, city)
                if db_price is not None:
                    total += db_price * meta.person_num
                else:
                    df = self.db.get_attractions(city) if city else pd.DataFrame()
                    if not df.empty:
                        matches = df[df["attraction_name"].str.lower() == name.lower()]
                        if not matches.empty:
                            ticket = matches.iloc[0].get("ticket_price", 0)
                            if pd.notna(ticket):
                                total += float(ticket) * meta.person_num
        
        for car in plan_data["cars"]:
            car_type = car.get("type") or car.get("car_type")
            city = car.get("city")
            if car_type:
                db_price = self.db.get_car_price(car_type, city)
                if db_price is not None:
                    total += db_price * meta.days
                else:
                    price = car.get("price_per_day", 0)
                    if isinstance(price, (int, float)):
                        total += price * meta.days
        
        return total
    
    def score_d3_budget(self, plan_data: dict, meta: QueryMeta) -> tuple[float, float]:
        if meta.budget <= 0:
            return None, 0.0
        
        total_cost = self.compute_total_cost(plan_data, meta)
        delta_b = max(0, (total_cost - meta.budget) / meta.budget)
        score = math.exp(-self.beta * delta_b)
        
        return score, total_cost
    
    def score_d4_impossible(self, plan_data: dict, meta: QueryMeta) -> Optional[float]:
        is_feasible = plan_data["is_feasible"]
        
        if meta.impossible:
            if not is_feasible:
                has_hallucination = False
                
                for flight in plan_data["flights"]:
                    fn = flight.get("flight_number")
                    if fn and not self.db.verify_flight(fn):
                        has_hallucination = True
                        break
                
                if not has_hallucination:
                    for hotel in plan_data["hotels"]:
                        name = hotel.get("name")
                        city = hotel.get("city")
                        if name and city and not self.db.verify_hotel(name, city):
                            has_hallucination = True
                            break
                
                if not has_hallucination:
                    for attr in plan_data["attractions"]:
                        name = attr.get("name")
                        city = attr.get("city")
                        if name and city and not self.db.verify_attraction(name, city):
                            has_hallucination = True
                            break
                
                return 0.0 if has_hallucination else 1.0
            else:
                return 0.0
        else:
            if is_feasible:
                return 1.0
            else:
                return 0.0
    
    def score_d5_retry_robustness(self, tool_call_count: int, meta: QueryMeta) -> float:
        cities_count = meta.cities_count if hasattr(meta, 'cities_count') and meta.cities_count else 1
        
        if meta.req_flight:
            trip_type = meta.req_flight.get("trip_type", "one_way")
            if cities_count > 1:
                min_flight_calls = cities_count
            else:
                min_flight_calls = 1
        else:
            min_flight_calls = 0
        
        min_hotel_calls = 1 if meta.req_hotel else 0
        min_attraction_calls = 1 if meta.req_attraction else 0
        min_car_calls = 1 if meta.req_car else 0
        
        min_submit = 1
        
        min_calls = min_flight_calls + min_hotel_calls + min_attraction_calls + min_car_calls + min_submit
        min_calls = max(min_calls, 2)
        
        extra_calls = max(0, tool_call_count - min_calls)
        
        max_tool_calls = 15
        penalty_per_call = 150 / max_tool_calls
        
        efficiency = max(0, 100 - penalty_per_call * extra_calls) / 100
        
        return efficiency
    

    
    def score_b2_opening_hours(self, plan_data: dict) -> float:
        visits = plan_data["attractions"]
        if not visits:
            return None
        
        violations = 0
        for attr in visits:
            name = attr.get("name")
            city = attr.get("city")
            visit_start = attr.get("visit_start")
            
            if name and city and visit_start:
                open_hours = self.db.get_open_hours(name, city)
                if not is_time_in_range(visit_start, open_hours):
                    violations += 1
        
        return 1 - violations / len(visits)
    
    def score_b3_spatiotemporal(self, plan_data: dict) -> float:
        events = []
        
        plan = plan_data.get("plan", {})
        for day_key in sorted(plan.keys()):
            day_data = plan.get(day_key, {})
            if not isinstance(day_data, dict):
                continue
            
            flights = day_data.get("flights", [])
            if isinstance(flights, list):
                for f in flights:
                    if isinstance(f, dict) and f.get("departure_time"):
                        events.append({
                            "type": "flight",
                            "start": f.get("departure_time"),
                            "end": f.get("arrival_time"),
                            "city": f.get("arrival_city"),
                            "name": f.get("flight_number")
                        })
            
            attractions = day_data.get("attractions", [])
            if isinstance(attractions, list):
                for a in attractions:
                    if isinstance(a, dict) and a.get("visit_start"):
                        events.append({
                            "type": "attraction",
                            "start": a.get("visit_start"),
                            "end": a.get("visit_end"),
                            "city": a.get("city"),
                            "name": a.get("name")
                        })
            
            hotel = day_data.get("hotel")
            if isinstance(hotel, dict) and hotel.get("check_in"):
                events.append({
                    "type": "hotel",
                    "start": hotel.get("check_in"),
                    "end": None,
                    "city": hotel.get("city"),
                    "name": hotel.get("name")
                })
        
        if len(events) < 2:
            return None
        
        violations = 0
        for i in range(len(events) - 1):
            e1, e2 = events[i], events[i+1]
            
            end_time = parse_time(e1.get("end") or e1.get("start"))
            start_time = parse_time(e2.get("start"))
            
            if end_time is None or start_time is None:
                continue
            
            delta_t = start_time - end_time
            
            coord1 = self.db.get_coordinates(e1["name"], e1["type"], e1["city"])
            coord2 = self.db.get_coordinates(e2["name"], e2["type"], e2["city"])
            
            if coord1 and coord2:
                distance = haversine_distance(coord1[0], coord1[1], coord2[0], coord2[1])
                min_time = get_min_travel_time(distance)
                
                if delta_t < min_time:
                    violations += 1
        
        transitions = len(events) - 1
        return 1 - violations / transitions if transitions > 0 else 1.0
    
    def score_query(self, plan_output: dict, meta: QueryMeta, tool_call_count: int = 0) -> ScoreResult:
        plan_data = self.extract_plan_data(plan_output)
        
        result = ScoreResult(
            query_index=0,
            is_feasible=plan_data["is_feasible"],
            attraction_count=len(plan_data["attractions"])
        )
        
        if not meta.impossible and not plan_data["is_feasible"]:
            result.d0_keyword = 0.0
            result.d0_source = 0.0
            result.d1_implicit = 0.0
            if meta.cities_count == 1:
                result.d2_unord_single = 0.0
            else:
                result.d2_unord_multi = 0.0
            result.d3_budget = 0.0 if meta.budget > 0 else None
            result.d4_impossible = 0.0
            result.d5_retry = 0.0
            result.d5_retry = 0.0
            result.b2_opening_hours = 0.0
            result.b3_spatiotemporal = 0.0
            result.total_cost = 0.0
            return result
        
        if meta.impossible and plan_data["is_feasible"]:
            result.d0_keyword = 0.0
            result.d0_source = 0.0
            result.d1_implicit = 0.0
            if meta.cities_count == 1:
                result.d2_unord_single = 0.0
            else:
                result.d2_unord_multi = 0.0
            result.d3_budget = 0.0 if meta.budget > 0 else None
            result.d4_impossible = 0.0
            result.d5_retry = 0.0
            result.b2_opening_hours = 0.0
            result.b3_spatiotemporal = 0.0
            result.total_cost = 0.0
            return result
        
        # D0
        result.d0_keyword = self.score_d0_keyword(plan_data, meta)
        result.d0_source = self.score_d0_source(plan_data, meta)
        
        # D1
        result.d1_implicit = self.score_d1_implicit(plan_data, meta)
        
        # D2
        d2_scores = self.score_d2_city(plan_data, meta)
        result.d2_unord_single = d2_scores.get("d2_unord_single")
        result.d2_unord_multi = d2_scores.get("d2_unord_multi")
        
        # D3
        d3_score, total_cost = self.score_d3_budget(plan_data, meta)
        result.d3_budget = d3_score
        result.total_cost = total_cost
        
        # D4
        result.d4_impossible = self.score_d4_impossible(plan_data, meta)
        
        result.d5_retry = self.score_d5_retry_robustness(tool_call_count, meta)
        
        # CCR
        result.b2_opening_hours = self.score_b2_opening_hours(plan_data)
        result.b3_spatiotemporal = self.score_b3_spatiotemporal(plan_data)
        
        return result


def compute_aggregate_scores(results: list[ScoreResult]) -> dict:
    def avg_non_none(values):
        valid = [v for v in values if v is not None]
        return sum(valid) / len(valid) if valid else None
    
    agg = {
        "d0_keyword": avg_non_none([r.d0_keyword for r in results]),
        "d0_source": avg_non_none([r.d0_source for r in results]),
        "d1_implicit": avg_non_none([r.d1_implicit for r in results]),
        "d2_unord_single": avg_non_none([r.d2_unord_single for r in results]),
        "d2_unord_multi": avg_non_none([r.d2_unord_multi for r in results]),
        "d3_budget": avg_non_none([r.d3_budget for r in results]),
        "d4_impossible": avg_non_none([r.d4_impossible for r in results]),
        "d5_retry": avg_non_none([r.d5_retry for r in results]),
        "d5_retry": avg_non_none([r.d5_retry for r in results]),
        "b2_opening_hours": avg_non_none([r.b2_opening_hours for r in results]),
        "b3_spatiotemporal": avg_non_none([r.b3_spatiotemporal for r in results]),
    }
    
    d2_combined = avg_non_none([agg["d2_unord_single"], agg["d2_unord_multi"]])
    satisfaction_scores = [agg["d0_keyword"], agg["d1_implicit"], d2_combined, agg["d3_budget"]]
    cat_satisfaction = avg_non_none(satisfaction_scores)
    
    cat_truthfulness = agg["d0_source"]
    
    reasoning_scores = [agg["d4_impossible"], agg["b2_opening_hours"], agg["b3_spatiotemporal"]]
    cat_reasoning = avg_non_none(reasoning_scores)
    
    cat_efficiency = agg["d5_retry"]
    
    agg["cat_satisfaction"] = cat_satisfaction
    agg["cat_truthfulness"] = cat_truthfulness
    agg["cat_reasoning"] = cat_reasoning
    agg["cat_efficiency"] = cat_efficiency
    
    category_scores = [cat_satisfaction, cat_truthfulness, cat_reasoning, cat_efficiency]
    valid_cats = [(s, 0.25) for s in category_scores if s is not None]
    if valid_cats:
        weighted_sum = sum(s * w for s, w in valid_cats)
        total_weight = sum(w for _, w in valid_cats)
        agg["weighted_overall"] = weighted_sum / total_weight
    else:
        agg["weighted_overall"] = None
    
    all_scores = [v for k, v in agg.items() if v is not None and k.startswith(("d0", "d1", "d2", "d3", "d4", "d5", "b2", "b3"))]
    agg["overall_avg"] = sum(all_scores) / len(all_scores) if all_scores else None
    
    return agg


def process_single_query(args):
    i, result, meta, scorer = args
    
    query_idx = result.get("query_index", i)
    plan_output = result.get("plan", {})
    retry_count = result.get("tool_call_count", 0)
    
    try:
        score = scorer.score_query(plan_output, meta, retry_count)
        score.query_index = query_idx
        return score
    except Exception as e:
        print(f"Error scoring query {query_idx}: {e}")
        return None

def main():
    parser = argparse.ArgumentParser(description="KDD Travel Planning Scoring Script")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file (LLM pipeline output)")
    parser.add_argument("--meta", "-m", required=True, help="Query metadata CSV file")
    parser.add_argument("--output", "-o", help="Output scores CSV file")
    parser.add_argument("--beta", type=float, default=4.0, help="Budget decay coefficient (D3)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Overflow decay coefficient (B1)")
    parser.add_argument("--k-threshold", type=int, default=15, help="Attraction threshold (B1)")
    parser.add_argument("--workers", type=int, default=4, help="Number of concurrent workers")
    
    args = parser.parse_args()
    
    try:
        from tqdm import tqdm
        from concurrent.futures import ThreadPoolExecutor
    except ImportError:
        print("Please install tqdm: pip install tqdm")
        sys.exit(1)

    print(f"Loading LLM outputs from {args.input}...")
    with open(args.input, "r", encoding="utf-8") as f:
        llm_data = json.load(f)
    
    results_list = llm_data.get("results", [])
    print(f"Loaded {len(results_list)} results")
    
    print(f"Loading query metadata from {args.meta}...")
    queries = load_queries(args.meta)
    print(f"Loaded {len(queries)} queries")
    
    db = get_sandbox_db()
    
    print("Pre-loading embedding model for thread safety...")
    get_embedding_model()
    
    scorer = TravelPlanScorer(db, {
        "beta": args.beta,
        "alpha": args.alpha,
        "k_threshold": args.k_threshold
    })
    
    print(f"\nScoring with {args.workers} threads...")
    tasks = []
    for i, result in enumerate(results_list):
        query_idx = result.get("query_index", i)
        if query_idx >= len(queries):
            print(f"Warning: query_index {query_idx} out of range, skipping")
            continue
        
        meta = queries[query_idx]
        tasks.append((i, result, meta, scorer))
    
    score_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        results = list(tqdm(executor.map(process_single_query, tasks), total=len(tasks), unit="q"))
        
        score_results = [r for r in results if r is not None]
    
    print(f"Successfully scored {len(score_results)} queries")
    
    agg = compute_aggregate_scores(score_results)
    
    print("\n" + "=" * 50)
    print("AGGREGATE SCORES Code Updated")
    print("=" * 50)
    for key, value in agg.items():
        if value is not None:
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: N/A")
    
    if args.output:
        output_path = args.output
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = os.path.splitext(os.path.basename(args.input))[0]
        output_path = f"scores_{input_name}_{timestamp}.csv"
    
    df = pd.DataFrame([asdict(r) for r in score_results])
    df.to_csv(output_path, index=False)
    print(f"\nScores saved to {output_path}")
    
    agg_path = output_path.replace(".csv", "_aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump({
            "config": {
                "beta": args.beta,
                "alpha": args.alpha,
                "k_threshold": args.k_threshold
            },
            "input_file": args.input,
            "meta_file": args.meta,
            "total_queries": len(score_results),
            "aggregate_scores": agg
        }, f, ensure_ascii=False, indent=2)
    print(f"Aggregate scores saved to {agg_path}")


if __name__ == "__main__":
    main()
