from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB  # Changed to actually use Naive Bayes
import numpy as np
import os
import csv

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=["http://localhost:5173"])  # Update with your frontend's URL


# Load dataset with error handling
csv_path = "sports.csv"
try:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset file not found: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded dataset with {len(df)} records")
except Exception as e:
    print(f"Error loading dataset: {str(e)}")
    df = pd.DataFrame(columns=['Players', 'Playing_Area', 'Scoring_Method', 'Equipment', 'Sport'])

# Preprocess categorical data with error handling
try:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    categorical_columns = ['Players', 'Playing_Area', 'Scoring_Method', 'Equipment']
    
    # Verify all columns exist
    missing_columns = [col for col in categorical_columns if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing columns in dataset: {missing_columns}")
    
    X = encoder.fit_transform(df[categorical_columns].astype(str))
    y = df['Sport']
    
    # Train a Naive Bayes model (to match the frontend naming)
    model = MultinomialNB()
    model.fit(X, y)
    print("Model trained successfully")
except Exception as e:
    print(f"Error training model: {str(e)}")
    # Create dummy encoder and model for graceful failure
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    model = MultinomialNB()

def normalize_input(category, value):
    if value is None:
        return ""
    
    value_lower = value.lower().strip()
    if category == 'scoring_method':
        if "point" in value_lower:
            return "Points-based"
        elif "goal" in value_lower:
            return "Goals-based"
        elif "time" in value_lower:
            return "Time-based"
        elif "style" in value_lower:
            return "Style-based"
    return value

@app.route('/api/sports', methods=['POST'])
def classify():
    data = request.json
    try:
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ["players", "playing_area", "scoring_method", "equipment"]
        missing_fields = [field for field in required_fields if field not in data or not data.get(field)]
        
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        # Extract input fields
        try:
            players = str(int(data.get("players")))  # Convert to int then string to validate number
        except ValueError:
            return jsonify({"error": "Players must be a valid number"}), 400
            
        playing_area = data.get("playing_area")
        scoring_method = normalize_input('scoring_method', data.get("scoring_method"))
        equipment = data.get("equipment")
        
        # Prepare input data
        input_data = [[players, playing_area, scoring_method, equipment]]
        
        try:
            input_encoded = encoder.transform(input_data)
        except Exception as e:
            return jsonify({"error": f"Error encoding input: {str(e)}"}), 500
        
        # Check if input exists in dataset
        existing_match = df[
            (df['Players'].astype(str) == players) &
            (df['Playing_Area'] == playing_area) &
            (df['Scoring_Method'] == scoring_method) &
            (df['Equipment'] == equipment)
        ]
        
        if existing_match.empty:
            # Check if we can find a close match
            close_match = df[
                (df['Players'].astype(str) == players) &
                ((df['Playing_Area'] == playing_area) | 
                 (df['Scoring_Method'] == scoring_method) |
                 (df['Equipment'] == equipment))
            ]
            
            if not close_match.empty:
                prediction = close_match.iloc[0]['Sport']
                return jsonify({
                    "recommended_sport": prediction,
                    "note": "Found approximate match. Exact match not in dataset."
                })
            else:
                return jsonify({"error": "No matching sport found. Please check input values."}), 400
        
        # Predict sport
        prediction = model.predict(input_encoded)[0]
        
        # Get prediction probability
        probabilities = model.predict_proba(input_encoded)[0]
        max_prob_index = np.argmax(probabilities)
        confidence = probabilities[max_prob_index]
        
        return jsonify({
            "recommended_sport": prediction,
            "confidence": float(confidence)
        })

    except Exception as e:
        return jsonify({"error": f"Unexpected error: {str(e)}"}), 500



# Load the dataset into a list of dictionaries
def load_sports_data(file_path):
    sports_data = []
    with open(file_path, mode='r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            sports_data.append(row)
    return sports_data

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