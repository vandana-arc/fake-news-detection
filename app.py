from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import joblib  # To load your ML model
import json

app = Flask(__name__)
CORS(app)

# Simple mock model (replace with an actual trained model)
def load_model():
    # Load or mock load a pre-trained fake-news classifier
    return lambda x: "fake" if "fake" in x.lower() else "real"

model = load_model()

@app.route('/api/detect', methods=['POST'])
def detect_fake_news():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid input"}), 400
        
        text = data['text']
        # Predict using the model
        prediction = model(text)
        is_fake = prediction == "fake"
        
        # Response
        return jsonify({"is_fake": is_fake})
    
    except Exception as e:
        print(e)
        return jsonify({"error": "Something went wrong"}), 500

if __name__ == '__main__':
    app.run(debug=True)