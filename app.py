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
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Use environment variable for MongoDB URI
MONGO_URI = os.getenv('MONGODB_URI', 'mongodb+srv://dbUser:12345@cluster0.dgpab.mongodb.net/project11?retryWrites=true&w=majority&tls=true')

def connect_to_mongodb():
    """Establish MongoDB connection with error handling"""
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        # Test connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        raise

def find_amenities(x, places):
    """Find and join amenities for given places"""
    try:
        temp = places[places["_id"].isin([ObjectId(id) for id in x])]
        temp = temp[temp["amenities"] != ""]
        temp = temp["amenities"].to_list()
        return " ".join(sorted(temp))
    except Exception as e:
        logger.error(f"Error in find_amenities: {e}")
        return ""

def get_destinations_by_category(db, categories):
    """Fetch destinations grouped by category"""
    try:
        destinations_by_category = {}
        for category in categories:
            destinations = list(db.destinations.find(
                {
                    "category": category,
                    "status": "approved"
                },
                {
                    "_id": 1,
                    "category": 1,
                    "amenities": 1,
                    "destination_name": 1
                }
            ))
            destinations_by_category[category] = destinations
            logger.info(f"Found {len(destinations)} destinations for category {category}")
        return destinations_by_category
    except Exception as e:
        logger.error(f"Error fetching destinations by category: {e}")
        raise

def distribute_destinations(destinations_by_category, days, spots_per_day=5):
    """Distribute destinations across days ensuring category balance"""
    try:
        daily_recommendations = []
        categories = list(destinations_by_category.keys())
        
        # Create a copy of destinations to work with
        working_destinations = {
            cat: list(dests) for cat, dests in destinations_by_category.items()
        }

        for day in range(days):
            day_destinations = []
            spots_per_category = spots_per_day // len(categories)
            extra_spots = spots_per_day % len(categories)
            
            logger.info(f"Day {day + 1} - Allocating {spots_per_category} spots per category with {extra_spots} extra spots")

            # First pass: distribute evenly across categories
            for category in categories:
                available = working_destinations[category]
                spots = spots_per_category + (1 if extra_spots > 0 else 0)
                extra_spots = max(0, extra_spots - 1)

                if available:
                    shuffle(available)  # Randomize selection
                    selected = available[:spots]
                    day_destinations.extend(selected)
                    working_destinations[category] = available[spots:]

            # Second pass: fill any remaining spots
            remaining_spots = spots_per_day - len(day_destinations)
            while remaining_spots > 0:
                for category in categories:
                    if remaining_spots <= 0:
                        break
                    if working_destinations[category]:
                        day_destinations.append(working_destinations[category].pop(0))
                        remaining_spots -= 1

            # Add day's destinations to recommendations
            daily_recommendations.extend([str(dest["_id"]) for dest in day_destinations])
            logger.info(f"Day {day + 1} complete with {len(day_destinations)} destinations")

        return daily_recommendations
    except Exception as e:
        logger.error(f"Error in distribute_destinations: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }), 200

@app.route('/recommend', methods=['POST'])
def recommend():
    """Main recommendation endpoint"""
    start_time = datetime.now()
    logger.info("Starting recommendation process")
    
    try:
        # Validate request
        data = request.json
        user_id = data.get('user_id')
        categories = data.get('category')
        days = data.get('days')

        logger.info(f"Request received - User: {user_id}, Categories: {categories}, Days: {days}")

        # Validate input data
        if not all([user_id, categories, days]):
            return jsonify({
                "error": "Missing required parameters",
                "required": ["user_id", "category", "days"],
                "received": data
            }), 400
        if not isinstance(categories, list):
            return jsonify({"error": "Categories must be a list"}), 400
        if not isinstance(days, int) or days <= 0:
            return jsonify({"error": "Days must be a positive integer"}), 400

        # Connect to MongoDB
        client = connect_to_mongodb()
        db = client["project11"]

        # Get destinations by category
        destinations_by_category = get_destinations_by_category(db, categories)
        logger.info(f"Fetched destinations by category: {destinations_by_category}")

        # Check if destinations are available
        if not destinations_by_category:
            logger.warning("No destinations found for the selected categories")
            return jsonify({"error": "No destinations available for the selected categories"}), 404

        # Check if there are enough destinations
        total_destinations = sum(len(dests) for dests in destinations_by_category.values())
        spots_per_day = 5
        if total_destinations < days * spots_per_day:
            logger.warning(f"Insufficient destinations: {total_destinations} available, {days * spots_per_day} required")
            return jsonify({"error": "Insufficient destinations for the selected categories and days"}), 400

        # If user has no history, distribute destinations evenly
        if not list(db.saved_destinations.find({"user_id": user_id})):
            logger.info(f"No history for user {user_id}, using balanced distribution")
            recommendations = distribute_destinations(
                destinations_by_category,
                days
            )
            processing_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"Recommendations generated in {processing_time:.2f} seconds")
            return jsonify(recommendations)

        # Get user's saved destinations for personalization
        all_destinations = []
        for cat_dests in destinations_by_category.values():
            all_destinations.extend(cat_dests)

        places = pd.DataFrame(all_destinations)
        logger.info(f"Places DataFrame: {places}")

        # Process amenities
        places = places.apply(lambda x: x.astype(str).str.lower())
        places = places.apply(lambda x: x.astype(str).str.strip())
        places["amenities"] = places["amenities"].str.replace('.', '')
        places["amenities"] = places["amenities"].str.replace(" ", "")
        places["amenities"] = places["amenities"].apply(lambda x: x.split(","))
        places["amenities"] = places["amenities"].apply(lambda x: [item.strip() for item in x])
        places["amenities"] = places["amenities"].apply(lambda x: sorted(x))
        places["amenities"] = places["amenities"].apply(lambda x: " ".join(x))

        # Check if amenities are available
        if places["amenities"].empty:
            logger.warning("No amenities found for destinations")
            return jsonify({"error": "No amenities available for destinations"}), 404

        # Calculate similarities
        tfidf = TfidfVectorizer()
        place_tfidf = tfidf.fit_transform(places["amenities"])
        logger.info(f"TF-IDF matrix shape: {place_tfidf.shape}")

        # Get personalized recommendations
        recommendations = distribute_destinations(
            destinations_by_category,
            days
        )

        processing_time = (datetime.now() - start_time).total_seconds()
        logger.info(f"Recommendations generated in {processing_time:.2f} seconds")
        
        return jsonify(recommendations)

    except Exception as e:
        logger.error(f"Error in recommendation process: {e}")
        return jsonify({
            "error": "Internal server error",
            "message": str(e)
        }), 500
    
    finally:
        if 'client' in locals():
            client.close()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)