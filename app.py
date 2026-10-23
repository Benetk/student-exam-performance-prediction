import json
import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request, render_template

app = Flask(__name__)
MODEL_PATH = os.environ.get("MODEL_PATH", "artifacts/model.joblib")
METADATA_PATH = os.environ.get("METADATA_PATH", "artifacts/metadata.json")

def load_artifacts():
    if not os.path.isfile(MODEL_PATH) or not os.path.isfile(METADATA_PATH):
        return None, None
    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
        model = joblib.load(MODEL_PATH)
        features = metadata.get("features")
        if not isinstance(features, list) or not features:
            return None, None
        return model, metadata
    except Exception:
        return None, None

@app.get("/")
def home():
    return render_template("index.html")

@app.get("/health")
def health():
    return jsonify({"status": "ok"}), 200

@app.get("/ready")
def ready():
    model, metadata = load_artifacts()
    if model is None:
        return jsonify({"ready": False, "reason": "Model artifacts missing or incompatible"}), 503
    return jsonify({"ready": True}), 200

@app.get("/model-info")
def model_info():
    model, metadata = load_artifacts()
    if model is None:
        return jsonify({"error": "Model artifacts missing or incompatible"}), 503
    return jsonify(metadata), 200

@app.post("/predict")
def predict():
    model, metadata = load_artifacts()
    if model is None:
        return jsonify({"error": "Model artifacts missing or incompatible"}), 503
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        return jsonify({"error": "Request body must be a JSON object"}), 400

    features = metadata["features"]
    missing = [name for name in features if name not in payload]
    if missing:
        return jsonify({"error": "Missing fields", "fields": missing}), 400

    schema = {item["name"]: item for item in metadata.get("input_schema", [])}
    clean = {}
    for name in features:
        spec = schema.get(name, {})
        value = payload[name]
        if spec.get("type") == "number":
            try:
                number = float(value)
            except (TypeError, ValueError):
                return jsonify({"error": f"{name} must be numeric"}), 400
            if number < float(spec["min"]) or number > float(spec["max"]):
                return jsonify({"error": f"{name} must be between {spec['min']} and {spec['max']}"}), 400
            clean[name] = number
        else:
            choices = spec.get("choices", [])
            value = str(value)
            if value not in choices:
                return jsonify({"error": f"{name} must be one of the allowed options", "choices": choices}), 400
            clean[name] = value

    try:
        frame = pd.DataFrame([clean], columns=features)
        prediction = float(model.predict(frame)[0])
        prediction = max(0.0, min(20.0, prediction))
    except Exception:
        return jsonify({"error": "Prediction failed due to model input incompatibility"}), 500
    return jsonify({
        "predicted_grade": round(prediction, 2),
        "scale": "0–20",
        "message": "This is a model estimate, not a guaranteed result."
    }), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
