from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os

# Load environment variables (optional)
SECRET_KEY = os.getenv("SECRET_KEY")
DB_URL = os.getenv("DATABASE_URL")

# Initialize Flask app
app = Flask(__name__)

# Check if model file exists
MODEL_PATH = "house_price_predictor.pkl"
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Please upload it.")

# Load the trained model
model = joblib.load(MODEL_PATH)

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert JSON data to DataFrame
        new_data = pd.DataFrame(data)

        # Predict house price
        predicted_price = model.predict(new_data)

        return jsonify({'predicted_price': f"${predicted_price[0]:,.2f}"})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400  # Return error with 400 Bad Request status

# Run the Flask app (production mode)
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)  # Set port to match Render logs
