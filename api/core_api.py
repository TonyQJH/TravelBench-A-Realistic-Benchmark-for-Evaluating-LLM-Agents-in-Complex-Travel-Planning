# This code is for API calling in Section 3.2

import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import ast




CUR_DIR = os.path.dirname(os.path.abspath(__file__))
ATTRACTION_CSV_DIR = os.path.normpath(os.path.join(CUR_DIR, 'data', 'attraction_data'))
FAC_GROUP_EMB_DIR = os.path.join(ATTRACTION_CSV_DIR, 'facilities_group_embedding')
HOTEL_CSV_DIR = os.path.normpath(os.path.join(CUR_DIR, 'data', 'hotel_data'))
AMEN_GROUP_EMB_DIR = os.path.join(HOTEL_CSV_DIR, 'amenities_group_embedding')
CAR_CSV_DIR = os.path.normpath(os.path.join(CUR_DIR, 'data', 'car_data'))
EXTRA_SERVICES_GROUP_EMB_DIR = os.path.join(CAR_CSV_DIR, 'extra_services_group_embedding')
FLIGHT_CSV_PATH = os.path.join(CUR_DIR, 'data', 'flight_data', 'flights.csv')

embedding_model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

def parse_str_list(val):
    """
    Convert a string-represented list or a real list to a Python list.

    Args:
        val (str or list): Value to parse.
    Returns:
        list: Parsed list, or empty list if parsing fails.
    """
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        try:
            v = ast.literal_eval(val)
            if isinstance(v, list):
                return v
        except Exception:
            pass
    return []

def search_attractions_by_struct_and_group(
    city_name,
    attraction_name=None,
    open_hours=None,
    max_ticket_price=None,
    ticket_price=None, # Legacy alias for max
    min_ticket_price=None,
    max_duration_of_visit=None,
    duration_of_visit=None, # Legacy alias for max
    min_duration_of_visit=None,
    min_rate_of_restaurant=None,
    rate_of_restaurant=None, # Legacy alias for min
    max_rate_of_restaurant=None,
    facilities_group=None,
    sort_by=None,
    sort_order="asc",
    top_k=10
):
    """
    Search for attractions in a city with structured filters and group-based facility embedding matching.
    Arguments accept explicit min/max prefixes. Legacy arguments (ticket_price, etc.) are supported as aliases.
    """
    final_max_price = max_ticket_price if max_ticket_price is not None else ticket_price
    final_max_duration = max_duration_of_visit if max_duration_of_visit is not None else duration_of_visit
    final_min_rate = min_rate_of_restaurant if min_rate_of_restaurant is not None else rate_of_restaurant

    city_std = city_name.strip().title()
    csv_path = os.path.join(ATTRACTION_CSV_DIR, f"{city_std}_attraction.csv")
    emb_path = os.path.join(FAC_GROUP_EMB_DIR, f"{city_std}_facilities_group.npy")
    if not os.path.exists(csv_path):
        return {"error": "csv not found"}
    df = pd.read_csv(csv_path)

    if attraction_name:
        name_lc = str(attraction_name).lower().strip()
        df = df[df["attraction_name"].str.lower().str.contains(name_lc, na=False)]

    if open_hours:
        def time_to_minutes(time_str):
            h, m = map(int, time_str.strip().split(":"))
            return h * 60 + m
        try:
            param_minute = time_to_minutes(open_hours)
            def check_time(row):
                if pd.isnull(row["open_hours"]): return False
                try:
                    open_str, close_str = row["open_hours"].split('-')
                    open_m = time_to_minutes(open_str)
                    close_m = time_to_minutes(close_str)
                    return open_m <= param_minute <= close_m
                except:
                    return False
            df = df[df.apply(check_time, axis=1)]
        except Exception:
            return {"error": "open_hours should be HH:MM"}

    def extract_price(s):
        if isinstance(s, str):
            s = s.replace(",", "")
            parts = s.strip().split(" ")[0]
            try:
                return float(parts)
            except:
                return np.nan
        try:
            return float(s)
        except:
            return np.nan

    if final_max_price is not None:
        try:
            ticket_val = float(final_max_price)
            df["ticket_price_num"] = df["ticket_price"].map(extract_price)
            df = df[df["ticket_price_num"] <= ticket_val]
        except Exception:
            return {"error": "max_ticket_price should be a number"}

    if min_ticket_price is not None:
        try:
            min_val = float(min_ticket_price)
            if "ticket_price_num" not in df.columns:
                df["ticket_price_num"] = df["ticket_price"].map(extract_price)
            df = df[df["ticket_price_num"] >= min_val]
        except Exception:
            return {"error": "min_ticket_price should be a number"}

    def get_duration(row):
        try:
            if isinstance(row["duration_of_visit"], str):
                return float(row["duration_of_visit"].split()[0])
            return float(row["duration_of_visit"])
        except:
            return 9999

    if final_max_duration is not None:
        try:
            param_duration = float(final_max_duration)
            df = df[df.apply(lambda row: get_duration(row) <= param_duration, axis=1)]
        except Exception:
            return {"error": "max_duration_of_visit should be a number hour"}

    if min_duration_of_visit is not None:
        try:
            min_duration = float(min_duration_of_visit)
            df = df[df.apply(lambda row: get_duration(row) >= min_duration, axis=1)]
        except Exception:
            return {"error": "min_duration_of_visit should be a number hour"}

    if final_min_rate is not None:
        try:
            param_rate = float(final_min_rate)
            df = df[df["rate_of_restaurant"].astype(float) >= param_rate]
        except Exception:
            return {"error": "min_rate_of_restaurant should be a number"}

    if max_rate_of_restaurant is not None:
        try:
            max_rate = float(max_rate_of_restaurant)
            df = df[df["rate_of_restaurant"].astype(float) <= max_rate]
        except Exception:
            return {"error": "max_rate_of_restaurant should be a number"}

    if sort_by and len(df) > 0:
        sort_field_map = {
            "ticket_price": "ticket_price_num",
            "price": "ticket_price_num",
            "rate_of_restaurant": "rate_of_restaurant",
            "rating": "rate_of_restaurant",
            "duration_of_visit": "duration_of_visit"
        }
        if sort_by in sort_field_map:
            field = sort_field_map[sort_by]
            if field == "ticket_price_num" and "ticket_price_num" not in df.columns:
                df["ticket_price_num"] = df["ticket_price"].map(extract_price)
            ascending = (sort_order.lower() == "asc")
            df = df.sort_values(by=field, ascending=ascending, na_position='last')

    
    results = []
    if facilities_group and len(df) > 0:
        if not os.path.exists(emb_path):
            return {"error": "facilities_group embedding not found, please generate first"}
        group_embeddings = np.load(emb_path)
        
        group_embeddings = group_embeddings[df.index]
        query_emb = embedding_model.encode([facilities_group])
        dim = group_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(group_embeddings)
        D, I = index.search(query_emb, min(top_k, len(df)))
        result_df = df.iloc[I[0]]
    else:
        result_df = df.head(top_k)

    final_keys = [
        "attraction_id","city_name","attraction_name","address","longitude","latitude",
        "open_hours","ticket_price","overview","facilities","type",
        "duration_of_visit","rate_of_restaurant"
    ]

    results = [
        {k: row.get(k, "") for k in final_keys}
        for row in result_df.to_dict(orient="records")
    ]
    for r in results:
        if "facilities" in r:
            r["facilities"] = parse_str_list(r["facilities"])
    return results




def search_hotels_by_struct_and_group(
    city,
    name=None,
    max_price=None,
    price=None, # Legacy alias for max
    min_price=None,
    min_rating=None,
    rating=None, # Legacy alias for min
    max_rating=None,
    min_star=None,
    star=None, # Legacy alias for min
    max_star=None,
    min_rate_of_restaurant=None,
    rate_of_restaurant=None, # Legacy alias for min
    amenities_group=None,
    sort_by=None,
    sort_order="asc",
    top_k=10
):
    """
    Search for hotels in a city with structured filters and group-based amenities embedding matching.
    Arguments accept explicit min/max prefixes. Legacy arguments (price, rating, etc.) are supported as aliases.
    """
    final_max_price = max_price if max_price is not None else price
    final_min_rating = min_rating if min_rating is not None else rating
    final_min_star = min_star if min_star is not None else star
    final_min_rate = min_rate_of_restaurant if min_rate_of_restaurant is not None else rate_of_restaurant

    city_std = city.strip().title()
    csv_path = os.path.join(HOTEL_CSV_DIR, f"{city_std}_hotel.csv")
    emb_path = os.path.join(AMEN_GROUP_EMB_DIR, f"{city_std}_amenities_group.npy")
    if not os.path.exists(csv_path):
        return {"error": "csv not found"}
    df = pd.read_csv(csv_path)

    if name:
        name_lc = str(name).lower().strip()
        df = df[df["name"].str.lower().str.contains(name_lc, na=False)]

    if final_max_price is not None:
        try:
            param_price = float(final_max_price)
            df = df[df["price"].astype(float) <= param_price]
        except Exception:
            return {"error": "max_price should be a number"}

    if min_price is not None:
        try:
            min_val = float(min_price)
            df = df[df["price"].astype(float) >= min_val]
        except Exception:
            return {"error": "min_price should be a number"}

    if final_min_rating is not None:
        try:
            param_rating = float(final_min_rating)
            df = df[df["rating"].astype(float) >= param_rating]
        except Exception:
            return {"error": "min_rating should be a number"}

    if max_rating is not None:
        try:
            max_rat = float(max_rating)
            df = df[df["rating"].astype(float) <= max_rat]
        except Exception:
            return {"error": "max_rating should be a number"}

    if final_min_star is not None:
        try:
            param_star = float(final_min_star)
            df = df[df["star"].astype(float) >= param_star]
        except Exception:
            return {"error": "min_star should be a number"}

    if max_star is not None:
        try:
            max_s = float(max_star)
            df = df[df["star"].astype(float) <= max_s]
        except Exception:
            return {"error": "max_star should be a number"}

    if final_min_rate is not None:
        try:
            param_rate = float(final_min_rate)
            df = df[df["rate_of_restaurant"].astype(float) >= param_rate]
        except Exception:
            return {"error": "min_rate_of_restaurant should be a number"}

    if sort_by and len(df) > 0:
        sort_field_map = {
            "price": "price",
            "rating": "rating",
            "star": "star"
        }
        if sort_by in sort_field_map:
            field = sort_field_map[sort_by]
            ascending = (sort_order.lower() == "asc")
            df = df.sort_values(by=field, ascending=ascending, na_position='last')

    
    results = []
    if amenities_group and len(df) > 0:
        if not os.path.exists(emb_path):
            return {"error": "amenities_group embedding not found, please generate first"}
        group_embeddings = np.load(emb_path)
        group_embeddings = group_embeddings[df.index]
        query_emb = embedding_model.encode([amenities_group])
        dim = group_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(group_embeddings)
        D, I = index.search(query_emb, min(top_k, len(df)))
        result_df = df.iloc[I[0]]
    else:
        result_df = df.head(top_k)

    final_keys = [
        "hotel_id","city_name","name","address","price","star","rating",
        "rate_of_restaurant","longitude","latitude",
        "about","amenities"
    ]
    results = [
        {k: row.get(k, "") for k in final_keys}
        for row in result_df.to_dict(orient="records")
    ]
    for r in results:
        if "amenities" in r:
            r["amenities"] = parse_str_list(r["amenities"])
    return results

def search_rental_cars_by_struct_and_group(
    city_name,
    max_price_per_day=None,
    price_per_day=None, # Legacy alias for max
    min_price_per_day=None,
    car_type=None,
    min_capacity=None,
    capacity=None, # Legacy alias for min
    max_capacity=None,
    extra_services_group=None,
    sort_by=None,
    sort_order="asc",
    top_k=10
):
    """
    Search for rental cars in a city with structured filters and group-based extra services embedding matching.
    Arguments accept explicit min/max prefixes. Legacy arguments (price_per_day, etc.) are supported as aliases.
    """
    final_max_price = max_price_per_day if max_price_per_day is not None else price_per_day
    final_min_capacity = min_capacity if min_capacity is not None else capacity

    city_std = city_name.strip().title()
    csv_path = os.path.join(CAR_CSV_DIR, f"{city_std}_rental_cars.csv")
    emb_path = os.path.join(EXTRA_SERVICES_GROUP_EMB_DIR, f"{city_std}_extra_services_group.npy")
    if not os.path.exists(csv_path):
        return {"error": "csv not found"}
    df = pd.read_csv(csv_path)

    if final_max_price is not None:
        try:
            price_val = float(final_max_price)
            df = df[df["price_per_day"].astype(float) <= price_val]
        except Exception:
            return {"error": "max_price_per_day should be a number"}

    if min_price_per_day is not None:
        try:
            min_val = float(min_price_per_day)
            df = df[df["price_per_day"].astype(float) >= min_val]
        except Exception:
            return {"error": "min_price_per_day should be a number"}

    if car_type:
        ct = str(car_type).lower().strip()
        df = df[df["car_type"].str.lower().str.contains(ct, na=False)]

    if final_min_capacity is not None:
        try:
            cap_val = int(final_min_capacity)
            df = df[df["capacity"].astype(int) >= cap_val]
        except Exception:
            return {"error": "min_capacity should be an integer"}

    if max_capacity is not None:
        try:
            max_cap = int(max_capacity)
            df = df[df["capacity"].astype(int) <= max_cap]
        except Exception:
            return {"error": "max_capacity should be an integer"}

    if sort_by and len(df) > 0:
        sort_field_map = {
            "price_per_day": "price_per_day",
            "price": "price_per_day",
            "capacity": "capacity"
        }
        if sort_by in sort_field_map:
            field = sort_field_map[sort_by]
            ascending = (sort_order.lower() == "asc")
            df = df.sort_values(by=field, ascending=ascending, na_position='last')

    
    if extra_services_group and len(df) > 0:
        if not os.path.exists(emb_path):
            return {"error": "extra_services_group embedding not found, please generate first"}
        group_embeddings = np.load(emb_path)
        group_embeddings = group_embeddings[df.index]
        query_emb = embedding_model.encode([extra_services_group])
        dim = group_embeddings.shape[1]
        index = faiss.IndexFlatL2(dim)
        index.add(group_embeddings)
        D, I = index.search(query_emb, min(top_k, len(df)))
        result_df = df.iloc[I[0]]
    else:
        result_df = df.head(top_k)

   
    final_keys = [
        "car_id","city_name", "price_per_day", "pickup_location", "car_type", "capacity",
        "extra_services"
    ]
    results = [
        {k: row.get(k, "") for k in final_keys}
        for row in result_df.to_dict(orient="records")
    ]
    for r in results:
        if "extra_services" in r:
            r["extra_services"] = parse_str_list(r["extra_services"])
    return results

def search_flights(
    departure_city,
    arrival_city,
    trip_type,
    max_price=None,
    price=None, # Legacy alias for max
    min_price=None,
    sort_by=None,
    sort_order="asc",
    top_k=10
):
    """
    Search for flights between two cities with support for one-way and round-trip, filtered by price.
    Arguments accept explicit min/max prefixes. Legacy arguments (price) are supported as aliases.
    """
    final_max_price = max_price if max_price is not None else price

    dep_city_std = departure_city.strip().title()
    arr_city_std = arrival_city.strip().title()
    
    csv_path = FLIGHT_CSV_PATH
    if not os.path.exists(csv_path):
        return {"error": "flights.csv not found"}
    df = pd.read_csv(csv_path)

    columns_to_keep = [
        "flight_id", "departure_city", "arrival_city", "departure_airport_name", "arrival_airport_name",
        "departure_time", "arrival_time", "flight_number", "price", 
        "departure_airport_latitude", "departure_airport_longitude",
        "arrival_airport_latitude", "arrival_airport_longitude"
    ]

    def apply_filters_and_sort(dataframe):
        if final_max_price is not None:
            try:
                p = float(final_max_price)
                dataframe = dataframe[dataframe["price"].astype(float) <= p]
            except:
                pass 
        if min_price is not None:
            try:
                min_p = float(min_price)
                dataframe = dataframe[dataframe["price"].astype(float) >= min_p]
            except:
                pass

        if sort_by and len(dataframe) > 0:
            field = sort_by
            if sort_by == "departure_time":
                field = "departure_time"
            elif sort_by == "price":
                field = "price"
            
            ascending = (sort_order.lower() == "asc")
            dataframe = dataframe.sort_values(by=field, ascending=ascending)
        
        return dataframe, None

  
    if trip_type == "one_way":
        out_flights = df[
            (df["departure_city"].str.lower().str.strip() == dep_city_std.lower()) &
            (df["arrival_city"].str.lower().str.strip() == arr_city_std.lower())
        ]
        
        out_flights, error = apply_filters_and_sort(out_flights)
        if error:
            return error
            
        result = (
            out_flights[columns_to_keep]
            .head(top_k)
            .to_dict(orient="records")
        )
        return {"flights": result}

 
    elif trip_type == "round_trip":
        out_flights = df[
            (df["departure_city"].str.lower().str.strip() == dep_city_std.lower()) &
            (df["arrival_city"].str.lower().str.strip() == arr_city_std.lower())
        ]
        ret_flights = df[
            (df["departure_city"].str.lower().str.strip() == arr_city_std.lower()) &
            (df["arrival_city"].str.lower().str.strip() == dep_city_std.lower())
        ]
        
        out_flights, error = apply_filters_and_sort(out_flights)
        if error:
            return error
        ret_flights, error = apply_filters_and_sort(ret_flights)
        if error:
            return error
            
        result = {
            "depart_flights": (
                out_flights[columns_to_keep]
                .head(top_k)
                .to_dict(orient="records")
            ),
            "return_flights": (
                ret_flights[columns_to_keep]
                .head(top_k)
                .to_dict(orient="records")
            ),
        }
        return result

    else:
        return {"error": "trip_type should be one_way or round_trip"}
