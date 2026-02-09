"""
KDD Travel Planning - Implicit Requirements Scoring Module (D1)
================================================================


"""

import os
import ast
import json
import numpy as np
import pandas as pd
from typing import Optional, Any
from dataclasses import dataclass

try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
API_DIR = os.path.join(CUR_DIR, "api")
DATA_DIR = os.path.join(API_DIR, "data")
CONFIG_DIR = os.path.join(CUR_DIR, "new_travelbench")

HOTEL_DATA_DIR = os.path.join(DATA_DIR, "hotel_data")
ATTRACTION_DATA_DIR = os.path.join(DATA_DIR, "attraction_data")
CAR_DATA_DIR = os.path.join(DATA_DIR, "car_data")

SEMANTIC_SIMILARITY_THRESHOLD = 0.65
EMBEDDING_MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        if not HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError("sentence-transformers is required. Install with: pip install sentence-transformers")
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _embedding_model


HOTEL_FACILITY_KEYWORDS = {
    "with children": ["Child-friendly rooms", "Baby cots", "Stroller storage", "Diaper changing tables", "Nursing rooms"],
    "road trip": ["Parking", "24-hour front desk", "EV charging for electric vehicles"],
    "elderly travelers": ["Elevators", "Bathroom grab bars", "Medical contacts", "Breakfast with dietary options"],
    "business travelers": ["High speed WiFi", "Business center", "Meeting rooms", "Laundry service", "Print machine"],
    "with pets": ["Pet-friendly", "Pet rest area", "Pet beds", "Durable floors"],
    "nightlife enthusiast": ["24-hour front desk", "Late room service", "On-site bars", "Rooftop lounges/nightclubs"],
    "disabled traveler": ["Accessible entrances", "Ramps/lifts", "Accessible rooms/bathrooms", "Braille/raised signage"],
    "fast-paced budget travel": ["Free Wi-Fi", "Communal kitchens", "Laundry", "Lockers"],
    "couples trip": ["Bathtubs/jacuzzis", "Scenic rooms/villas", "Spa"],
    "solo women": ["Women-only floors/rooms", "Privacy-conscious check-in", "24-hour security", "Surveillance cameras", "Double locks on doors"],
    "luxury travelers": ["Spa", "Gym", "Premium bedding", "Bar"],
    "foodie": [],
    "photography": ["Scenic rooms", "Sunrise calls", "Photography services/packages"],
}

ATTRACTION_FACILITY_KEYWORDS = {
    "with children": ["family restrooms", "Nursing facilities"],
    "road trip": ["parking"],
    "elderly travelers": ["wheelchair rental", "benches/rest areas"],
    "business travelers": ["High speed WiFi"],
    "with pets": ["Pet-friendly", "pet water stations", "Pet rest area"],
    "nightlife enthusiast": ["night markets", "bars", "evening shows"],
    "disabled traveler": ["ramps", "elevators", "wheelchair rentals", "accessible restrooms"],
    "fast-paced budget travel": ["city passes", "luggage storage", "self-guided tours"],
    "couples trip": ["scenic spots", "sunset cruises", "special couples experiences"],
    "solo women": ["group tours", "security"],
    "luxury travelers": ["private tours", "skip the line passes", "VIP exclusive events"],
    "foodie": ["Food markets", "tasting tours"],
    "photography": ["photo spots", "guided photo tours", "charging stations"],
}

CAR_FACILITY_KEYWORDS = {
    "with children": ["Child seats", "Child locks"],
    "road trip": ["Roadside emergency support"],
    "elderly travelers": ["Advanced Driving Assistance Systems"],
    "business travelers": ["WiFi"],
    "with pets": ["Pet seat belts", "Pet-friendly"],
    "nightlife enthusiast": ["Late pick up services"],
    "disabled traveler": ["Wheelchair space", "Hand controls", "Accessible cars"],
    "fast-paced budget travel": [],
    "couples trip": [],
    "solo women": ["Women-only cars", "Airport safe waiting areas"],
    "luxury travelers": ["First-class car", "Chauffeured cars", "Guide/driver package"],
    "foodie": [],
    "photography": ["Remote shoots"],
}

SPECIAL_RULES = {
    "luxury travelers": {
        "hotel": {"dual_dimension": True, "star_threshold": 5}
    },
    "foodie": {
        "hotel": {"rating_only": True, "restaurant_rating_threshold": 4.0},
        "attraction": {"dual_dimension": True, "restaurant_rating_threshold": 3.5}
    }
}


class ImplicitScoringDB:
    
    def __init__(self):
        self._hotels_cache: dict[str, pd.DataFrame] = {}
        self._attractions_cache: dict[str, pd.DataFrame] = {}
        self._cars_cache: dict[str, pd.DataFrame] = {}
    
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
    
    def get_hotel_info(self, name: str, city: str) -> Optional[dict]:
        df = self.get_hotels(city)
        if df.empty:
            return None
        matches = df[df["name"].str.lower() == name.lower()]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return {
            "name": row.get("name"),
            "amenities": self._parse_list(row.get("amenities", "[]")),
            "star": float(row.get("star", 0)),
            "rating": float(row.get("rating", 0)),
            "restaurant_rating": float(row.get("rate_of_restaurant", 0)),
        }
    
    def get_attraction_info(self, name: str, city: str) -> Optional[dict]:
        df = self.get_attractions(city)
        if df.empty:
            return None
        matches = df[df["attraction_name"].str.lower() == name.lower()]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return {
            "name": row.get("attraction_name"),
            "facilities": self._parse_list(row.get("facilities", "[]")),
            "restaurant_rating": float(row.get("rate_of_restaurant", 0)) if pd.notna(row.get("rate_of_restaurant")) else 0,
        }
    
    def get_car_info(self, car_type: str, city: str) -> Optional[dict]:
        df = self.get_cars(city)
        if df.empty:
            return None
        matches = df[df["car_type"].str.lower() == car_type.lower()]
        if matches.empty:
            matches = df[df["car_type"].str.lower().str.contains(car_type.lower(), regex=False)]
        if matches.empty:
            return None
        row = matches.iloc[0]
        return {
            "car_type": row.get("car_type"),
            "extra_services": self._parse_list(row.get("extra_services", "[]")),
        }
    
    def _parse_list(self, val) -> list:
        if pd.isna(val) or val == "":
            return []
        try:
            if isinstance(val, list):
                return val
            return ast.literal_eval(str(val))
        except:
            return []


class ImplicitRequirementsScorer:
    
    def __init__(self, db: ImplicitScoringDB = None, similarity_threshold: float = None):
        self.db = db or ImplicitScoringDB()
        self.similarity_threshold = similarity_threshold or SEMANTIC_SIMILARITY_THRESHOLD
        self._embedding_model = None
        self._embedding_cache = {}
    
    def _get_model(self):
        if self._embedding_model is None:
            self._embedding_model = get_embedding_model()
        return self._embedding_model
    
    def _get_embedding(self, text: str) -> np.ndarray:
        if text not in self._embedding_cache:
            model = self._get_model()
            self._embedding_cache[text] = model.encode([text])[0]
        return self._embedding_cache[text]
    
    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def check_facility_match_semantic(self, resource_facilities: list, keywords: list) -> float:
        """
        
        Args:
        
        Returns:
        """
        if not keywords or not resource_facilities:
            return 0.0
        
        facility_text = ", ".join(resource_facilities)
        facility_emb = self._get_embedding(facility_text)
        
        keyword_text = ", ".join(keywords)
        keyword_emb = self._get_embedding(keyword_text)
        
        similarity = self._cosine_similarity(facility_emb, keyword_emb)
        
        if similarity >= self.similarity_threshold:
            score = 100.0
        else:
            score = (similarity / self.similarity_threshold) * 100
        
        return min(100.0, max(0.0, score))
    
    def check_facility_match(self, resource_facilities: list, keywords: list) -> bool:
        """
        """
        if not keywords:
            return False
        score = self.check_facility_match_semantic(resource_facilities, keywords)
        return score > 0
    
    def score_hotel(self, hotel_info: dict, implicit_keyword: str) -> Optional[float]:
        """
        """
        kw_lower = implicit_keyword.lower()
        keywords = HOTEL_FACILITY_KEYWORDS.get(kw_lower, [])
        special = SPECIAL_RULES.get(kw_lower, {}).get("hotel", {})
        
        amenities = hotel_info.get("amenities", [])
        star = hotel_info.get("star", 0)
        restaurant_rating = hotel_info.get("restaurant_rating", 0)
        
        if special.get("dual_dimension"):
            facility_score = self.check_facility_match_semantic(amenities, keywords)
            star_score = 100 if star >= special.get("star_threshold", 5) else 0
            return (facility_score + star_score) / 2
        
        if special.get("rating_only"):
            threshold = special.get("restaurant_rating_threshold", 4.0)
            return 100 if restaurant_rating >= threshold else 0
        
        if not keywords:
            return None
        
        return self.check_facility_match_semantic(amenities, keywords)
    
    def score_attraction(self, attraction_info: dict, implicit_keyword: str) -> Optional[float]:
        """
        """
        kw_lower = implicit_keyword.lower()
        keywords = ATTRACTION_FACILITY_KEYWORDS.get(kw_lower, [])
        special = SPECIAL_RULES.get(kw_lower, {}).get("attraction", {})
        
        facilities = attraction_info.get("facilities", [])
        restaurant_rating = attraction_info.get("restaurant_rating", 0)
        
        if special.get("dual_dimension"):
            facility_score = self.check_facility_match_semantic(facilities, keywords)
            rating_score = 100 if restaurant_rating >= special.get("restaurant_rating_threshold", 3.5) else 0
            return (facility_score + rating_score) / 2
        
        if not keywords:
            return None
        
        return self.check_facility_match_semantic(facilities, keywords)
    
    def score_car(self, car_info: dict, implicit_keyword: str) -> Optional[float]:
        """
        """
        kw_lower = implicit_keyword.lower()
        keywords = CAR_FACILITY_KEYWORDS.get(kw_lower, [])
        
        if not keywords:
            return None
        
        extra_services = car_info.get("extra_services", [])
        return self.check_facility_match_semantic(extra_services, keywords)
    
    def score_plan(self, plan_data: dict, implicit_keywords: list) -> dict:
        """
        
        Args:
        
        Returns:
            {
                "total_score": float (0-1),
                "details": {
                    "hotel_scores": [...],
                    "attraction_scores": [...],
                    "car_scores": [...],
                }
            }
        """
        if not implicit_keywords:
            return {"total_score": None, "details": {}}
        
        hotels = plan_data.get("hotels", [])
        attractions = plan_data.get("attractions", [])
        cars = plan_data.get("cars", [])
        cities = plan_data.get("cities", [])
        
        all_scores = []
        details = {
            "hotel_scores": [],
            "attraction_scores": [],
            "car_scores": [],
        }
        
        for hotel in hotels:
            hotel_name = hotel.get("name", "")
            hotel_info = None
            for city in cities:
                hotel_info = self.db.get_hotel_info(hotel_name, city)
                if hotel_info:
                    break
            
            if not hotel_info:
                continue
            
            for kw in implicit_keywords:
                score = self.score_hotel(hotel_info, kw)
                if score is not None:
                    all_scores.append(score)
                    details["hotel_scores"].append({
                        "hotel": hotel_name,
                        "keyword": kw,
                        "score": score
                    })
        
        for attraction in attractions:
            attr_name = attraction.get("name", "")
            city = attraction.get("city", "")
            
            attraction_info = None
            if city:
                attraction_info = self.db.get_attraction_info(attr_name, city)
            if not attraction_info:
                for c in cities:
                    attraction_info = self.db.get_attraction_info(attr_name, c)
                    if attraction_info:
                        break
            
            if not attraction_info:
                continue
            
            for kw in implicit_keywords:
                score = self.score_attraction(attraction_info, kw)
                if score is not None:
                    all_scores.append(score)
                    details["attraction_scores"].append({
                        "attraction": attr_name,
                        "keyword": kw,
                        "score": score
                    })
        
        for car in cars:
            car_type = car.get("type") or car.get("car_type", "")
            city = car.get("city", "")
            
            car_info = None
            if city:
                car_info = self.db.get_car_info(car_type, city)
            if not car_info:
                for c in cities:
                    car_info = self.db.get_car_info(car_type, c)
                    if car_info:
                        break
            
            if not car_info:
                continue
            
            for kw in implicit_keywords:
                score = self.score_car(car_info, kw)
                if score is not None:
                    all_scores.append(score)
                    details["car_scores"].append({
                        "car": car_type,
                        "keyword": kw,
                        "score": score
                    })
        
        if all_scores:
            total_score = sum(all_scores) / len(all_scores) / 100
        else:
            total_score = None
        
        return {
            "total_score": total_score,
            "details": details,
            "num_checks": len(all_scores),
            "avg_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        }


_scorer: Optional[ImplicitRequirementsScorer] = None

def get_implicit_scorer() -> ImplicitRequirementsScorer:
    global _scorer
    if _scorer is None:
        _scorer = ImplicitRequirementsScorer()
    return _scorer


def score_d1_implicit_v2(plan_data: dict, implicit_keywords: list) -> Optional[float]:
    """
    
    Args:
    
    Returns:
    """
    if not implicit_keywords:
        return None
    
    scorer = get_implicit_scorer()
    result = scorer.score_plan(plan_data, implicit_keywords)
    return result["total_score"]


if __name__ == "__main__":
    db = ImplicitScoringDB()
    scorer = ImplicitRequirementsScorer(db)
    
    hotel_info = db.get_hotel_info("Yan Garden Chaoyang", "Beijing")
    if hotel_info:
        print(f"Name: {hotel_info['name']}")
        print(f"Star: {hotel_info['star']}")
        print(f"Amenities: {hotel_info['amenities'][:5]}...")
        print(f"Restaurant Rating: {hotel_info['restaurant_rating']}")
        
        test_keywords = ["luxury travelers", "with children", "foodie", "road trip"]
        for kw in test_keywords:
            score = scorer.score_hotel(hotel_info, kw)
            print(f"  {kw}: {score}")
    
    mock_plan = {
        "hotels": [{"name": "Yan Garden Chaoyang"}],
        "attractions": [],
        "cars": [],
        "cities": ["Beijing"]
    }
    result = scorer.score_plan(mock_plan, ["luxury travelers", "road trip"])
    print(f"Total Score: {result['total_score']}")
    print(f"Num Checks: {result['num_checks']}")
    print(f"Details: {json.dumps(result['details'], indent=2)}")
