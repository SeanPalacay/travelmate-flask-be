from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import randint, shuffle
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

# Use environment variable for MongoDB URI
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://dbUser:12345@cluster0.dgpab.mongodb.net/project11?retryWrites=true&w=majority&tls=true')

def find_amenities(x, places):
    temp = places[places["_id"].isin([ObjectId(id) for id in x])]
    temp = temp[temp["amenities"] != ""]
    temp = temp["amenities"].to_list()
    return " ".join(sorted(temp))

def distribute_destinations(destinations, categories, days, spots_per_day=5):
    """
    Distribute destinations across days ensuring category balance
    """
    # Initialize daily recommendations
    daily_recommendations = []
    destinations_by_category = {cat: [] for cat in categories}
    
    # Group destinations by category
    for dest in destinations:
        cat = dest["category"].lower()
        if cat in destinations_by_category:
            destinations_by_category[cat].append(str(dest["_id"]))
    
    # Shuffle destinations within each category
    for cat in destinations_by_category:
        shuffle(destinations_by_category[cat])
    
    # Calculate spots per category per day
    total_spots = days * spots_per_day
    min_spots_per_category = total_spots // len(categories)
    
    # Distribute spots across days
    for day in range(days):
        day_spots = []
        remaining_spots = spots_per_day
        
        # First pass: try to get equal distribution
        spots_this_round = remaining_spots // len(categories)
        for cat in categories:
            if destinations_by_category[cat]:
                spots_to_take = min(spots_this_round, len(destinations_by_category[cat]))
                day_spots.extend(destinations_by_category[cat][:spots_to_take])
                destinations_by_category[cat] = destinations_by_category[cat][spots_to_take:]
                remaining_spots -= spots_to_take
        
        # Second pass: fill remaining spots
        while remaining_spots > 0:
            for cat in categories:
                if remaining_spots <= 0:
                    break
                if destinations_by_category[cat]:
                    day_spots.append(destinations_by_category[cat].pop(0))
                    remaining_spots -= 1
        
        # If still not enough spots, fill from any category
        while len(day_spots) < spots_per_day:
            for cat in categories:
                if len(day_spots) >= spots_per_day:
                    break
                if destinations_by_category[cat]:
                    day_spots.append(destinations_by_category[cat].pop(0))
        
        daily_recommendations.extend(day_spots)
    
    return daily_recommendations

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    categories = data.get('category')  # List of categories
    days = data.get('days')

    if not user_id or not categories or not days:
        return jsonify({"error": "user_id, category, and days are required"}), 400

    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client["project11"]
    destinations = db["destinations"]
    saved_destinations = db["saved_destinations"]

    # Get all relevant destinations
    all_destinations = list(destinations.find(
        {"category": {"$in": categories}, "status": "approved"},
        {"_id": 1, "category": 1, "amenities": 1}
    ))

    if not all_destinations:
        return jsonify({"error": "No destinations found for selected categories"}), 404

    # Get user's saved destinations for personalization
    user_saved = list(saved_destinations.find({"user_id": user_id}, {"destination_id": 1}))
    saved_ids = [ObjectId(doc["destination_id"]) for doc in user_saved]

    if not saved_ids:  # If user has no saved destinations
        # Randomly distribute destinations across days
        recommendations = distribute_destinations(all_destinations, categories, days)
        return jsonify(recommendations)

    # Prepare data for TF-IDF
    places = pd.DataFrame(all_destinations)
    places = places.apply(lambda x: x.astype(str).str.lower())
    places = places.apply(lambda x: x.astype(str).str.strip())

    # Process amenities
    places["amenities"] = places["amenities"].str.replace('.', '')
    places["amenities"] = places["amenities"].str.replace(" ", "")
    places["amenities"] = places["amenities"].apply(lambda x: x.split(","))
    places["amenities"] = places["amenities"].apply(lambda x: [item.strip() for item in x])
    places["amenities"] = places["amenities"].apply(lambda x: sorted(x))
    places["amenities"] = places["amenities"].apply(lambda x: " ".join(x))

    # Create user profile
    user_profile = pd.DataFrame({
        "_id": [user_id],
        "savedDestinations": [saved_ids]
    })
    user_profile["preferredAmenities"] = user_profile["savedDestinations"].apply(
        lambda x: find_amenities(x, places)
    )

    # Calculate similarities
    tfidf = TfidfVectorizer()
    place_tfidf = tfidf.fit_transform(places["amenities"])
    user_tfidf = tfidf.transform(user_profile["preferredAmenities"])
    cosine_sim = cosine_similarity(user_tfidf, place_tfidf)

    # Get similarity scores and sort
    sim_scores = list(enumerate(cosine_sim[-1, :]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Create recommendations list based on similarities
    recommended_destinations = []
    seen_ids = set(str(id) for id in saved_ids)
    
    for score in sim_scores:
        idx = score[0]
        if idx < len(places):
            dest_id = str(places.iloc[idx]["_id"])
            if dest_id not in seen_ids:
                dest_category = places.iloc[idx]["category"]
                if dest_category.lower() in [cat.lower() for cat in categories]:
                    recommended_destinations.append({
                        "_id": dest_id,
                        "category": dest_category
                    })
                    seen_ids.add(dest_id)

    # Distribute recommendations across days
    final_recommendations = distribute_destinations(
        recommended_destinations, 
        categories, 
        days
    )

    return jsonify(final_recommendations)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)