from flask import Flask, request, jsonify
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from bson import ObjectId
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from random import randint
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

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    category = data.get('category')
    days = data.get('days')

    if not user_id or not category or not days:
        return jsonify({"error": "user_id, category, and days are required"}), 400

    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
    db = client["project11"]
    destinations = db["destinations"]
    saved_destinations = db["saved_destinations"]

    data = list(destinations.find({}, {"_id": 1, "category": 1, "amenities": 1}))
    places = pd.DataFrame(data)

    places = places.apply(lambda x: x.astype(str).str.lower())
    places = places.apply(lambda x: x.astype(str).str.strip())

    places["amenities"] = places["amenities"].str.replace('.', '')
    places["amenities"] = places["amenities"].str.replace(" ", "")
    places["amenities"] = places["amenities"].apply(lambda x: x.split(","))
    places["amenities"] = places["amenities"].apply(lambda x: [item.strip() for item in x])
    places["amenities"] = places["amenities"].apply(lambda x: sorted(x))
    places["amenities"] = places["amenities"].apply(lambda x: " ".join(x))

    user_saved_destinations = list(saved_destinations.find({"user_id": user_id}, {"destination_id": 1}))
    user_saved_destinations = [ObjectId(doc["destination_id"]) for doc in user_saved_destinations]

    user_profile = pd.DataFrame({
        "_id": [user_id],
        "savedDestinations": [user_saved_destinations]
    })

    user_profile["preferredAmenities"] = user_profile["savedDestinations"].apply(lambda x: find_amenities(x, places))

    user = user_profile.loc[user_profile["_id"] == user_id]

    if user.empty:
        if places.shape[0] >= 10:
            recomend_list = []
            while len(recomend_list) < (5 * days):
                indx = randint(0, places.shape[0] - 1)
                if places["category"].iloc[indx] in category:
                    recomend_list.append(str(places["_id"].iloc[indx]))
            return jsonify(list(set(recomend_list)))
        else:
            return jsonify([str(id) for id in places["_id"].to_list()])

    tfidf = TfidfVectorizer()
    place_tfidf = tfidf.fit_transform(places["amenities"])
    user_tfidf = tfidf.transform(user_profile["preferredAmenities"])
    cosine_sim = cosine_similarity(user_tfidf, place_tfidf)

    sim_scores = list(enumerate(cosine_sim[-1, :]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    i = 0
    recomend_list = [str(id) for id in user["savedDestinations"].iloc[0]]
    while len(recomend_list) < (5 * days):
        if i >= len(sim_scores):
            break
        indx = sim_scores[i][0]
        if places["category"].iloc[indx] in category:
            if str(places["_id"].iloc[indx]) not in recomend_list:
                recomend_list.append(str(places["_id"].iloc[indx]))
        i += 1

    return jsonify(recomend_list)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)  # Disable debug mode in production