import os
from flask import Flask, request, jsonify, Response
import markdown
from core_api import (
    search_attractions_by_struct_and_group,
    search_hotels_by_struct_and_group,
    search_rental_cars_by_struct_and_group,
    search_flights
)

app = Flask(__name__)

CUR_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CUR_DIR)
API_DOC_CANDIDATES = [
    os.path.join(CUR_DIR, "api_doc.md"),
    os.path.join(PARENT_DIR, "api_doc.md"),
    os.path.join(PARENT_DIR, "data", "api_doc.md"),
]

def find_api_doc():
    for path in API_DOC_CANDIDATES:
        if os.path.exists(path):
            return path
    return None

@app.route("/")
def home():
    doc_path = find_api_doc()
    if doc_path is None:
        return Response("<h1>API Documentation</h1><p>API documentation not found.</p>", mimetype="text/html", status=404)
    with open(doc_path, "r", encoding="utf-8") as f:
        content = f.read()
    html = markdown.markdown(content)
    return Response(html, mimetype="text/html")

@app.route("/attractions1", methods=["GET"])
def F_list_attractions():
    return jsonify({
        "message": "This API has been deprecated. Please use other attractions endpoints."
    }), 410

from flask import request, jsonify

@app.route("/attractions2", methods=["GET"])
def api_search_attractions2():
    allowed_params = {
        "city",
        "attraction_name",
        "open_hours",
        "ticket_price",
        "max_ticket_price",
        "min_ticket_price",
        "duration_of_visit",
        "max_duration_of_visit",
        "min_duration_of_visit",
        "rate_of_restaurant",
        "min_rate_of_restaurant",
        "max_rate_of_restaurant",
        "facility",
        "sort_by",
        "sort_order",
        "top_k"
    }

    args = request.args
    bad_keys = [k for k in args.keys() if k not in allowed_params]
    if bad_keys:
        return jsonify({
            "error": f"Unsupported parameter(s): {', '.join(bad_keys)}",
            "allowed_parameters": sorted(list(allowed_params))
        }), 400

    city_name = args.get("city")
    if not city_name or city_name.strip() == "":
        return jsonify({"error": "Parameter 'city' is required."}), 400

    result = search_attractions_by_struct_and_group(
        city_name=city_name,
        attraction_name=args.get("attraction_name"),
        open_hours=args.get("open_hours"),
        max_ticket_price=args.get("max_ticket_price"),
        ticket_price=args.get("ticket_price"),
        min_ticket_price=args.get("min_ticket_price"),
        max_duration_of_visit=args.get("max_duration_of_visit"),
        duration_of_visit=args.get("duration_of_visit"),
        min_duration_of_visit=args.get("min_duration_of_visit"),
        min_rate_of_restaurant=args.get("min_rate_of_restaurant"),
        rate_of_restaurant=args.get("rate_of_restaurant"),
        max_rate_of_restaurant=args.get("max_rate_of_restaurant"),
        facilities_group=args.get("facility"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 20)),
    )
    return jsonify(result)



@app.route("/hotels1", methods=["GET"])
def F_list_hotels():
    return jsonify({
        "message": "This API has been deprecated. Please use other hotels endpoints."
    }), 410

@app.route("/hotels2", methods=["GET"])
def api_search_hotels2():
    allowed_params = {
        "city",
        "name",
        "price",
        "max_price",
        "min_price",
        "rating",
        "min_rating",
        "max_rating",
        "star",
        "min_star",
        "max_star",
        "rate_of_restaurant",
        "min_rate_of_restaurant",
        "amenity",
        "sort_by",
        "sort_order",
        "top_k"
    }

    args = request.args
    bad_keys = [k for k in args.keys() if k not in allowed_params]
    if bad_keys:
        return jsonify({
            "error": f"Unsupported parameter(s): {', '.join(bad_keys)}",
            "allowed_parameters": sorted(list(allowed_params))
        }), 400

    city = args.get("city")
    if not city or city.strip() == "":
        return jsonify({"error": "Parameter 'city' is required."}), 400

    result = search_hotels_by_struct_and_group(
        city=city,
        name=args.get("name"),
        max_price=args.get("max_price"),
        price=args.get("price"),
        min_price=args.get("min_price"),
        min_rating=args.get("min_rating"),
        rating=args.get("rating"),
        max_rating=args.get("max_rating"),
        min_star=args.get("min_star"),
        star=args.get("star"),
        max_star=args.get("max_star"),
        min_rate_of_restaurant=args.get("min_rate_of_restaurant"),
        rate_of_restaurant=args.get("rate_of_restaurant"),
        amenities_group=args.get("amenity"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 20)),
    )
    return jsonify(result)




@app.route("/cars1", methods=["GET"])
def F_list_rental_cars():   
    return jsonify({
        "message": "This API has been deprecated. Please use other rental cars endpoints."
    }), 410

@app.route("/cars2", methods=["GET"])
def api_search_rental_cars2():
    allowed_params = {
        "city",
        "capacity",
        "min_capacity",
        "max_capacity",
        "price_per_day",
        "max_price_per_day",
        "min_price_per_day",
        "car_type",
        "extra_service",
        "sort_by",
        "sort_order",
        "top_k"
    }

    args = request.args
    bad_keys = [k for k in args.keys() if k not in allowed_params]
    if bad_keys:
        return jsonify({
            "error": f"Unsupported parameter(s): {', '.join(bad_keys)}",
            "allowed_parameters": sorted(list(allowed_params))
        }), 400

    city = args.get("city")
    capacity = args.get("capacity")
    # min_capacity is alias for capacity if capacity is present, logic handled in core but we must extract it safely
    # For cars, 'capacity' was required. We should allow 'min_capacity' as well.
    # Current logic: capacity is mandatory param in this func. Let's relax it if min_capacity is there.
    min_capacity = args.get("min_capacity")
    
    # Logic Update: Either capacity OR min_capacity should be present (or both if user mixes)
    # But effectively they mean the same (min requirement).
    effective_capacity = capacity if capacity else min_capacity

    if not city or city.strip() == "":
        return jsonify({"error": "Parameter 'city' is required."}), 400
    # Allowed to be None (Optional)
    # if not effective_capacity... logic removed to make it optional consistent with docs

    result = search_rental_cars_by_struct_and_group(
        city_name=city,
        max_price_per_day=args.get("max_price_per_day"),
        price_per_day=args.get("price_per_day"),
        min_price_per_day=args.get("min_price_per_day"),
        car_type=args.get("car_type"),
        min_capacity=args.get("min_capacity"),
        capacity=args.get("capacity"),
        max_capacity=args.get("max_capacity"),
        extra_services_group=args.get("extra_service"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 20)),
    )
    return jsonify(result)



@app.route("/flights1", methods=["GET"])
def F_list_flights():
    return jsonify({
        "message": "This API has been deprecated. Please use other flights endpoints."
    }), 410


@app.route("/flights2", methods=["GET"])
def api_search_flight2():
    allowed_params = {
        "departure_city",
        "arrival_city",
        "trip_type",
        "price",
        "max_price",
        "min_price",
        "sort_by",
        "sort_order",
        "top_k"
    }

    args = request.args
    bad_keys = [k for k in args.keys() if k not in allowed_params]
    if bad_keys:
        return jsonify({
            "error": f"Unsupported parameter(s): {', '.join(bad_keys)}",
            "allowed_parameters": sorted(list(allowed_params))
        }), 400

    departure_city = args.get("departure_city")
    arrival_city = args.get("arrival_city")
    trip_type = args.get("trip_type")

    if not all([departure_city, arrival_city, trip_type]):
         return jsonify({"error": "Parameters 'departure_city', 'arrival_city', 'trip_type' are required."}), 400

    result = search_flights(
        departure_city=departure_city,
        arrival_city=arrival_city,
        trip_type=trip_type,
        max_price=args.get("max_price"),
        price=args.get("price"),
        min_price=args.get("min_price"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 20)),
    )
    return jsonify(result)




import random

EMPTY_RESULT_CITIES = {
    "Timbuktu", "Ulaanbaatar", "Thimphu", "Vaduz", "Andorra la Vella",
    "San Marino", "Djibouti", "Belmopan", "Nukualofa", "Funafuti",
    "Yaren", "Palikir", "Majuro", "Tarawa", "Apia"
}

@app.route("/hotels_primary", methods=["GET"])
def api_hotels_primary():
    args = request.args
    city = args.get("city", "")
    
    if city in EMPTY_RESULT_CITIES or any(c.lower() in city.lower() for c in EMPTY_RESULT_CITIES):
        return jsonify({
            "status": "success",
            "message": f"No hotels found in {city}. Consider using hotels_backup for broader coverage.",
            "data": [],
            "total": 0
        })
    
    result = search_hotels_by_struct_and_group(
        city=city,
        name=args.get("name"),
        max_price=args.get("max_price"),
        min_price=args.get("min_price"),
        min_rating=args.get("min_rating"),
        min_star=args.get("min_star"),
        amenities_group=args.get("amenity"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 10)),
    )
    return jsonify({"status": "success", "data": result, "total": len(result)})


@app.route("/hotels_backup", methods=["GET"])
def api_hotels_backup():
    args = request.args
    city = args.get("city", "")
    
    if city in EMPTY_RESULT_CITIES or any(c.lower() in city.lower() for c in EMPTY_RESULT_CITIES):
        mock_hotels = [
            {
                "name": f"Grand Hotel {city}",
                "city": city,
                "price": 120 + random.randint(0, 80),
                "rating": round(3.5 + random.random() * 1.5, 1),
                "star": random.choice([3, 4, 5]),
                "amenities": ["wifi", "parking", "restaurant"]
            },
            {
                "name": f"Budget Inn {city}",
                "city": city,
                "price": 50 + random.randint(0, 30),
                "rating": round(3.0 + random.random(), 1),
                "star": random.choice([2, 3]),
                "amenities": ["wifi"]
            }
        ]
        return jsonify({"status": "success", "data": mock_hotels, "total": len(mock_hotels), "source": "backup"})
    
    result = search_hotels_by_struct_and_group(
        city=city,
        name=args.get("name"),
        max_price=args.get("max_price"),
        min_price=args.get("min_price"),
        min_rating=args.get("min_rating"),
        min_star=args.get("min_star"),
        amenities_group=args.get("amenity"),
        sort_by=args.get("sort_by"),
        sort_order=args.get("sort_order", "asc"),
        top_k=int(args.get("top_k", 10)),
    )
    return jsonify({"status": "success", "data": result, "total": len(result), "source": "backup"})


@app.route("/search_mixed", methods=["GET"])
def api_search_mixed():
    
    """
    args = request.args
    city = args.get("city", "")
    requested_type = args.get("type", "hotel")
    
    if not city or city.strip() == "":
        return jsonify({"error": "Parameter 'city' is required."}), 400
    
    return_wrong_type = random.random() < 0.5
    
    if requested_type == "hotel":
        if return_wrong_type:
            result = search_attractions_by_struct_and_group(
                city_name=city,
                top_k=int(args.get("top_k", 5)),
            )
            return jsonify({
                "status": "success",
                "requested_type": "hotel",
                "actual_type": "attraction",
                "data": result,
                "warning": "Data type mismatch. You requested hotels but received attractions."
            })
        else:
            result = search_hotels_by_struct_and_group(
                city=city,
                max_price=args.get("max_price"),
                min_rating=args.get("min_rating"),
                top_k=int(args.get("top_k", 5)),
            )
            return jsonify({
                "status": "success",
                "requested_type": "hotel",
                "actual_type": "hotel",
                "data": result
            })
    else:  # requested_type == "attraction"
        if return_wrong_type:
            result = search_hotels_by_struct_and_group(
                city=city,
                top_k=int(args.get("top_k", 5)),
            )
            return jsonify({
                "status": "success",
                "requested_type": "attraction",
                "actual_type": "hotel",
                "data": result,
                "warning": "Data type mismatch. You requested attractions but received hotels."
            })
        else:
            result = search_attractions_by_struct_and_group(
                city_name=city,
                top_k=int(args.get("top_k", 5)),
            )
            return jsonify({
                "status": "success",
                "requested_type": "attraction",
                "actual_type": "attraction",
                "data": result
            })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
