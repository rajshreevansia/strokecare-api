
import os
import joblib
import pandas as pd

from pathlib import Path
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load saved model files from this folder
BASE_DIR = Path(__file__).resolve().parent

model = joblib.load(
    BASE_DIR / "stroke_risk_tuned_random_forest.pkl"
)

preprocessor = joblib.load(
    BASE_DIR / "stroke_risk_preprocessor.pkl"
)


@app.route("/")
def home():
    return "Stroke Risk Prediction API is running!"


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({"error": "Missing JSON input"}), 400

        input_data = pd.DataFrame([{
            "age": float(data["age"]),
            "gender": data["gender"],
            "systolic": float(data["systolic"]),
            "distolic": float(data["diastolic"]),
            "glucose": float(data["glucose"]),
            "smoking": data["smoking"],
            "bmi": float(data["bmi"]),
            "cholestrol": float(data["cholesterol"])
        }])

        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)[0]

        return jsonify({"risk": int(prediction)})

    except (KeyError, TypeError, ValueError) as error:
        return jsonify({"error": str(error)}), 400


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
