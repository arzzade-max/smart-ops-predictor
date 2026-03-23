from flask import Flask, request, jsonify
from pathlib import Path
import pickle
import pandas as pd

app = Flask(__name__)

base_dir = Path(__file__).parent.parent
model_path = base_dir / "model.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]

    return jsonify({
        "prediction": int(prediction),
        "message": "High Risk" if prediction == 1 else "Low Risk"
    })

if __name__ == "__main__":
    app.run(debug=True)