import joblib
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('linear_regression_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return "Welcome to the House Price Prediction API! Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    # Expected input format:
    # {"Area": 1500, "Bedrooms": 3, "Bathrooms": 2, "Stories": 1, "Parking": 1}

    # Convert input to numpy array and reshape for prediction
    features = np.array([data['Area'], data['Bedrooms'], data['Bathrooms'], data['Stories'], data['Parking']]).reshape(1, -1)
    
    # Scale the features using the loaded scaler
    scaled_features = scaler.transform(features)
    
    # Make prediction
    prediction = model.predict(scaled_features)[0]
    
    return jsonify({'predicted_price': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
