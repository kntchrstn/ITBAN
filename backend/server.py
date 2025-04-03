from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.naive_bayes import MultinomialNB  # Changed to actually use Naive Bayes
import numpy as np
import os

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

@app.route('/api/prescriptive', methods=['POST'])
def prescriptive():
    data = request.json
    try:
        # Validate required fields
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        required_fields = ["players", "playing_area", "scoring_method"]
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({"error": f"Missing required fields: {', '.join(missing_fields)}"}), 400
        
        # Extract input fields with validation
        try:
            players = int(data.get("players", 0))
            if players <= 0:
                return jsonify({"error": "Number of players must be greater than 0"}), 400
        except (ValueError, TypeError):
            return jsonify({"error": "Players must be a valid number"}), 400
            
        playing_area = data.get("playing_area", "").lower()
        if not playing_area:
            return jsonify({"error": "Playing area cannot be empty"}), 400
            
        scoring_method = data.get("scoring_method", "").lower()
        if not scoring_method:
            return jsonify({"error": "Scoring method cannot be empty"}), 400
        
        # Initialize variables
        recommendations = []
        
        # Normalize inputs for better matching
        area_type = "indoor" if "indoor" in playing_area else "outdoor" if "outdoor" in playing_area else "water" if "water" in playing_area else "unknown"
        scoring_type = "points" if any(x in scoring_method for x in ["point", "score", "basket"]) else \
                       "goals" if any(x in scoring_method for x in ["goal", "net"]) else \
                       "time" if any(x in scoring_method for x in ["time", "speed", "fastest"]) else \
                       "runs" if any(x in scoring_method for x in ["run", "inning"]) else \
                       "style" if any(x in scoring_method for x in ["style", "judge", "rating"]) else "unknown"
        
        # Process according to number of players and normalized inputs
        # Individual sports (1-2 players)
        if players <= 2:
            if area_type == "indoor":
                if scoring_type == "points":
                    recommendations.append({
                        "sport": "Table Tennis",
                        "confidence": 0.95,
                        "explanation": "Table Tennis is perfect for 1-2 players in an indoor setting with a points-based system. It requires minimal equipment and space."
                    })
                    recommendations.append({
                        "sport": "Badminton",
                        "confidence": 0.90,
                        "explanation": "Badminton works well for singles or doubles in indoor courts with a points-based system. It offers fast-paced action in a controlled environment."
                    })
                    if players == 2:
                        recommendations.append({
                            "sport": "Squash",
                            "confidence": 0.85,
                            "explanation": "Squash is ideal for 2 players in an enclosed court with a points-based system. It's high-intensity and requires good reflexes."
                        })
                elif scoring_type == "time":
                    recommendations.append({
                        "sport": "Indoor Athletics",
                        "confidence": 0.85,
                        "explanation": "Indoor track and field events are perfect for time-based competitions in a controlled environment."
                    })
                elif scoring_type == "style":
                    recommendations.append({
                        "sport": "Dance Sport",
                        "confidence": 0.90,
                        "explanation": "Competitive dancing is judged on style and technique, making it perfect for 1-2 performers in an indoor setting."
                    })
            elif area_type == "water":
                if scoring_type == "time":
                    recommendations.append({
                        "sport": "Swimming",
                        "confidence": 0.98,
                        "explanation": "Swimming is the quintessential water sport for individuals, with races timed for precise performance measurement."
                    })
                    recommendations.append({
                        "sport": "Diving",
                        "confidence": 0.90,
                        "explanation": "Diving combines water sports with style-based scoring, ideal for individual competitors seeking technical challenges."
                    })
                elif scoring_type == "style":
                    recommendations.append({
                        "sport": "Synchronized Swimming",
                        "confidence": 0.95,
                        "explanation": "For 1-2 people in water with style-based scoring, synchronized swimming offers artistic expression and technical prowess."
                    })
            elif area_type == "outdoor":
                if scoring_type == "points":
                    recommendations.append({
                        "sport": "Tennis",
                        "confidence": 0.95,
                        "explanation": "Outdoor tennis is perfect for 1-2 players with its established points system and versatile outdoor courts."
                    })
                elif scoring_type == "time":
                    recommendations.append({
                        "sport": "Track and Field",
                        "confidence": 0.92,
                        "explanation": "Individual track events are time-based and perfect for outdoor competition."
                    })
                    recommendations.append({
                        "sport": "Cycling",
                        "confidence": 0.90,
                        "explanation": "Cycling races are timed events that can be enjoyed by individuals in outdoor settings."
                    })
                else:
                    recommendations.append({
                        "sport": "Golf",
                        "confidence": 0.95,
                        "explanation": "Golf is ideal for 1-2 players outdoors with its stroke-based scoring system and strategic gameplay."
                    })
        
        # Small team sports (3-6 players)
        elif players <= 6:
            if area_type == "indoor":
                if scoring_type == "points":
                    recommendations.append({
                        "sport": "Volleyball",
                        "confidence": 0.95,
                        "explanation": "Volleyball is perfectly suited for 6 players in an indoor court, featuring fast-paced points-based scoring and team coordination."
                    })
                    recommendations.append({
                        "sport": "Basketball",
                        "confidence": 0.92,
                        "explanation": "Basketball is an exciting format for 6 players indoors, with points scoring and reduced court size for faster gameplay."
                    })
                elif scoring_type == "goals":
                    recommendations.append({
                        "sport": "Indoor Hockey",
                        "confidence": 0.88,
                        "explanation": "Indoor hockey accommodates small teams with goals-based scoring in a controlled environment."
                    })
                    recommendations.append({
                        "sport": "Handball",
                        "confidence": 0.85,
                        "explanation": "Handball works well for 6 players indoors with its goals-based scoring system and dynamic gameplay."
                    })
            elif area_type == "water":
                recommendations.append({
                    "sport": "Water Polo",
                    "confidence": 0.90,
                    "explanation": "Water polo is ideal for small teams in water with a goals-based scoring system, combining swimming endurance with team strategy."
                })
                if scoring_type == "time":
                    recommendations.append({
                        "sport": "Swim Relay",
                        "confidence": 0.85,
                        "explanation": "Swim relays allow team competition in water with time-based scoring, combining individual skill with team coordination."
                    })
            elif area_type == "outdoor":
                if scoring_type == "goals":
                    recommendations.append({
                        "sport": "Football (5-a-side)",
                        "confidence": 0.95,
                        "explanation": "5-a-side football is perfect for small teams outdoors with goals-based scoring and requires less space than full-sized football."
                    })
                elif scoring_type == "points":
                    recommendations.append({
                        "sport": "Beach Volleyball",
                        "confidence": 0.92,
                        "explanation": "Beach volleyball is ideal for 4-6 players outdoors with a points-based system, offering a fun combination of skill and athleticism on sand."
                    })
                    recommendations.append({
                        "sport": "Ultimate Frisbee",
                        "confidence": 0.85,
                        "explanation": "Ultimate frisbee works well for small teams outdoors with a points-based system and minimal equipment requirements."
                    })
                elif scoring_type == "runs":
                    recommendations.append({
                        "sport": "Cricket (T10)",
                        "confidence": 0.80,
                        "explanation": "T10 cricket is a shortened format ideal for small teams with runs-based scoring outdoors."
                    })
        
        # Medium team sports (7-11 players)
        elif players <= 11:
            if area_type == "indoor":
                if scoring_type == "points":
                    recommendations.append({
                        "sport": "Basketball",
                        "confidence": 0.98,
                        "explanation": "Basketball is perfect for indoor teams up to 10 players (5v5), with its points-based scoring system and strategic gameplay."
                    })
                    recommendations.append({
                        "sport": "Volleyball",
                        "confidence": 0.95,
                        "explanation": "Volleyball accommodates up to 12 players (6v6) indoors with a points-based system emphasizing teamwork and coordination."
                    })
                elif scoring_type == "goals":
                    recommendations.append({
                        "sport": "Futsal",
                        "confidence": 0.90,
                        "explanation": "Futsal is ideal for indoor teams with goals-based scoring, offering a fast-paced alternative to outdoor football."
                    })
            elif area_type == "water":
                recommendations.append({
                    "sport": "Water Polo",
                    "confidence": 0.95,
                    "explanation": "Water polo accommodates medium-sized teams with 7 players per side, combining swimming with goals-based team strategy."
                })
            elif area_type == "outdoor":
                if scoring_type == "goals":
                    recommendations.append({
                        "sport": "Football",
                        "confidence": 0.99,
                        "explanation": "Football (soccer) is perfect for 11 players per team outdoors with goals-based scoring, offering the world's most popular team sport experience."
                    })
                    recommendations.append({
                        "sport": "Field Hockey",
                        "confidence": 0.90,
                        "explanation": "Field hockey accommodates 11 players per team outdoors with goals-based scoring and requires good team coordination and stick skills."
                    })
                elif scoring_type == "points":
                    recommendations.append({
                        "sport": "Rugby Sevens",
                        "confidence": 0.92,
                        "explanation": "Rugby Sevens features 7 players per team outdoors with a points-based system, offering a faster version of traditional rugby."
                    })
                    if players >= 9:
                        recommendations.append({
                            "sport": "Baseball",
                            "confidence": 0.85,
                            "explanation": "Baseball needs 9 players per team with its unique scoring system combining runs and outs in an outdoor field."
                        })
                elif scoring_type == "runs":
                    recommendations.append({
                        "sport": "Cricket",
                        "confidence": 0.95,
                        "explanation": "Cricket requires 11 players per team with its runs-based scoring system, offering strategic depth and various formats of play."
                    })
        
        # Large team sports (12+ players)
        else:
            if area_type == "indoor":
                recommendations.append({
                    "sport": "Indoor Cricket",
                    "confidence": 0.85,
                    "explanation": "Indoor cricket can accommodate larger teams with run-based scoring in a modified format suitable for indoor spaces."
                })
                recommendations.append({
                    "sport": "Korfball",
                    "confidence": 0.75,
                    "explanation": "Korfball is a mixed-gender sport for larger groups indoors with a points-based system similar to basketball but with unique rules."
                })
            elif area_type == "water":
                recommendations.append({
                    "sport": "Water Polo Tournament",
                    "confidence": 0.80,
                    "explanation": "A water polo tournament format works well for larger groups, allowing multiple teams to compete in water-based matches."
                })
                recommendations.append({
                    "sport": "Swimming Gala",
                    "confidence": 0.85,
                    "explanation": "Swimming galas accommodate many participants in various water-based events with time and points-based scoring systems."
                })
            elif area_type == "outdoor":
                if scoring_type == "points":
                    recommendations.append({
                        "sport": "Rugby Union",
                        "confidence": 0.95,
                        "explanation": "Rugby Union is designed for larger teams (15 players per side) with points-based scoring through tries, conversions, and penalties."
                    })
                    recommendations.append({
                        "sport": "American Football",
                        "confidence": 0.92,
                        "explanation": "American football accommodates larger teams with its points-based system through touchdowns, field goals, and other scoring methods."
                    })
                elif scoring_type == "goals":
                    recommendations.append({
                        "sport": "Lacrosse",
                        "confidence": 0.85,
                        "explanation": "Lacrosse works well for larger teams outdoors with goals-based scoring and combines elements of basketball, soccer, and hockey."
                    })
                elif scoring_type == "runs":
                    recommendations.append({
                        "sport": "Cricket Tournament",
                        "confidence": 0.90,
                        "explanation": "Cricket tournaments can accommodate many players across multiple teams with the traditional runs-based scoring system."
                    })
        
        # If no specific recommendations were found based on the rules
        if not recommendations:
            # Fallback recommendations
            if area_type == "water":
                recommendations.append({
                    "sport": "Swimming",
                    "confidence": 0.70,
                    "explanation": "Swimming is a versatile water sport that can be adapted for various numbers of participants and scoring systems."
                })
            elif area_type == "indoor":
                recommendations.append({
                    "sport": "Badminton",
                    "confidence": 0.70,
                    "explanation": "Badminton is a flexible indoor sport that can be played in singles or doubles with a points-based system."
                })
            else:
                recommendations.append({
                    "sport": "Athletics",
                    "confidence": 0.65,
                    "explanation": "Athletics offers a variety of events that can accommodate different numbers of participants and scoring preferences."
                })
        
        # Sort recommendations by confidence and take the top one as primary
        recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        primary_recommendation = recommendations[0]
        
        # Format response
        return jsonify({
            "recommendation": primary_recommendation["sport"],
            "explanation": primary_recommendation["explanation"],
            "confidence": primary_recommendation["confidence"],
            "alternatives": recommendations[1:3] if len(recommendations) > 1 else []
        })

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