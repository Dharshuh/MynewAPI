
from flask import Flask, request, jsonify
import joblib
import pandas as pd
import os
SECRET_KEY = os.getenv("SECRET_KEY")
DB_URL = os.getenv("DATABASE_URL")

# Load the trained model
model = joblib.load('house_price_predictor.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from request
    data = request.get_json()
    
    # Convert JSON data to DataFrame
    new_data = pd.DataFrame(data)
    
    # Predict house price
    predicted_price = model.predict(new_data)

    return jsonify({'predicted_price': f"${predicted_price[0]:,.2f}"})

if __name__ == '__main__':
    app.run(debug=True)
from flask import Flask, request, jsonify
import joblib
import pandas as pd

# Load the trained model
model = joblib.load('house_price_predictor.pkl')

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from request
        data = request.get_json()

        # Convert JSON data to DataFrame
        new_data = pd.DataFrame(data)

        # Predict house price
        predicted_price = model.predict(new_data)

        return jsonify({'predicted_price': f"${predicted_price[0]:,.2f}"})
    
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
