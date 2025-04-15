from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB

import csv

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Update with your frontend's URL

# Global variables
model = None
model_accuracy = None

# Load the dataset into a list of dictionaries
def load_sports_data(file_path):
    sports_data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sports_data.append(row)
    return sports_data


    global model_accuracy
    sports_data = pd.read_csv(file_path)

    # Preprocess data
    X = sports_data[['Players', 'Equipment', 'Playing_Area', 'Scoring_Method']]
    y = sports_data['Sport']

    # One-hot encode categorical features
    encoder = OneHotEncoder()
    X_encoded = encoder.fit_transform(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # Train the Naive Bayes model
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # Calculate accuracy
    y_pred = model.predict(X_test)
    model_accuracy = accuracy_score(y_test, y_pred)

    return model

# Rule-based classifier function
def predict_sport(sports_data, players, equipment, playing_area, scoring_method):
    for sport in sports_data:
        if (int(sport['Players']) == players and
            equipment.lower() in sport['Equipment'].lower() and
            playing_area.lower() in sport['Playing_Area'].lower() and
            scoring_method.lower() in sport['Scoring_Method'].lower()):
            return sport['Sport'], sport['Prescription']
    return None, "No matching sport found. Please refine your inputs."


@app.route('/api/prescriptive', methods=['POST'])
def prescriptive():
    data = request.json
    try:
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ["players", "playing_area", "scoring_method", "equipment"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        # Extract input fields
        try:
            players = int(data.get("players", 0))
            if players <= 0:
                return jsonify({"error": "Number of players must be greater than 0"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Players must be a valid number"}), 400
        
        playing_area = data.get("playing_area", "").lower()
        scoring_method = data.get("scoring_method", "").lower()
        equipment = data.get("equipment", "").lower()
        
        if not playing_area or not scoring_method or not equipment:
            return jsonify({"error": "Playing area, scoring method, and equipment cannot be empty"}), 400
        
        # Load sports data
        sports_data = load_sports_data("sports.csv")
        
        # Predict sport
        sport, prescription = predict_sport(sports_data, players, equipment, playing_area, scoring_method)
        
        if sport:
            return jsonify({
                "recommended_sport": sport,
                "prescription": prescription  # Ensure this field is included
            })
        else:
            return jsonify({"error": prescription}), 400

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        "status": "ok",
        "dataset_loaded": len(df) > 0,
        "model_trained": True
    })

if __name__ == '__main__':
    app.run(debug=True)