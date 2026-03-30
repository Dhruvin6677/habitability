import os
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from utils import prepare_input

app = Flask(__name__)
CORS(app)  # Allows frontend-backend communication

# Load ranked dataset if it exists
DATA_PATH = "data/habitability_ranked.csv"
ranked_data = pd.read_csv(DATA_PATH) if os.path.exists(DATA_PATH) else None

# Mock Model Class: Replaces the need for model.pkl
class HeuristicHabitModel:
    def __init__(self):
        # Features the frontend currently sends
        self.feature_names_in_ = [
            'koi_period', 'koi_prad', 'koi_teq', 'koi_insol', 
            'koi_smass', 'koi_slum', 'koi_steff', 'koi_srad'
        ]

    def predict_proba(self, df):
        """Calculates a probability based on Earth-similarity indices."""
        row = df.iloc[0]
        score = 0.5  # Base neutral score
        
        # Temperature Logic (Habitable roughly 180K - 310K)
        if 180 <= row['koi_teq'] <= 310: score += 0.2
        
        # Radius Logic (Earth-like is 0.5 to 2.0 Earth Radii)
        if 0.5 <= row['koi_prad'] <= 2.0: score += 0.15
        
        # Flux Logic (Earth = 1.0; HZ is roughly 0.25 to 1.77)
        if 0.25 <= row['koi_insol'] <= 1.77: score += 0.1
        
        prob = min(0.98, max(0.02, score))
        return [[1 - prob, prob]]

    def predict(self, df):
        """Returns 1 (Habitable) if probability > 0.5."""
        prob = self.predict_proba(df)[0][1]
        return [1] if prob > 0.5 else [0]

# Initialize the heuristic engine
model = HeuristicHabitModel()

@app.route('/')
def home():
    # Serves the index.html from the /templates folder
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        input_df = prepare_input(data, model)
        
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return jsonify({
            "prediction": int(prediction),
            "probability": float(probability),
            "status": "success"
        })
    except Exception as e:
        return jsonify({"error": str(e), "status": "failed"})

@app.route('/rank', methods=['GET'])
def rank():
    if ranked_data is None:
        return jsonify({"error": "Dataset not found"}), 404
    top = ranked_data.sort_values(by='habitability_score', ascending=False).head(10)
    return jsonify(top.to_dict(orient='records'))

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
