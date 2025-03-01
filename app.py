from flask import Flask, request, jsonify
from flask_cors import CORS  
import joblib
import pandas as pd
import os

app = Flask(__name__)
CORS(app)  

# Load model
MODEL_PATH = "house_price_predictor.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

# ✅ **Fix: Add a Home Route**
@app.route('/predict', methods=['POST'])

def home():
    return jsonify({"message": "API is live!"})

# ✅ **Fix: Handle Predictions Correctly**
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not found'}), 500

    try:
        data = request.get_json()
        
        if not isinstance(data, dict):
            return jsonify({'error': 'Invalid JSON format'}), 400

        new_data = pd.DataFrame([data])
        predicted_price = model.predict(new_data)

        return jsonify({'predicted_price': f"${predicted_price[0]:,.2f}"})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

# ✅ **Ensure app runs on Render's port**
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
