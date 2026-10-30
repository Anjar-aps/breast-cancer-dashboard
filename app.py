from pathlib import Path
import math
import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "breast_cancer_rf_model.joblib"
app = Flask(__name__)

FEATURE_LABELS = {
    "Clump_thickness": "Clump thickness",
    "Uniformity_of_cell_size": "Uniformity of cell size",
    "Uniformity_of_cell_shape": "Uniformity of cell shape",
    "Marginal_adhesion": "Marginal adhesion",
    "Single_epithelial_cell_size": "Single epithelial cell size",
    "Bare_nuclei": "Bare nuclei",
    "Bland_chromatin": "Bland chromatin",
    "Normal_nucleoli": "Normal nucleoli",
    "Mitoses": "Mitoses",
}

MODEL_PACKAGE = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None

@app.get("/")
def home():
    return render_template("index.html", model_ready=MODEL_PACKAGE is not None,
                           feature_labels=FEATURE_LABELS)

@app.get("/health")
def health():
    return jsonify({"status": "ok", "model_ready": MODEL_PACKAGE is not None})

@app.post("/api/predict")
def predict():
    if MODEL_PACKAGE is None:
        return jsonify({"error": "File model belum tersedia pada server."}), 503

    payload = request.get_json(silent=True) or {}
    features = MODEL_PACKAGE["feature_names"]
    missing = [f for f in features if f not in payload]
    if missing:
        return jsonify({"error": "Fitur belum lengkap.", "missing": missing}), 400

    values = {}
    for feature in features:
        try:
            value = float(payload[feature])
        except (TypeError, ValueError):
            return jsonify({"error": f"{feature} harus berupa angka."}), 400
        if not math.isfinite(value) or not 1 <= value <= 10:
            return jsonify({"error": f"{feature} harus berada pada rentang 1 sampai 10."}), 400
        values[feature] = value

    sample = pd.DataFrame([values], columns=features)
    pipeline = MODEL_PACKAGE["model"]
    prediction = int(pipeline.predict(sample)[0])
    probabilities = pipeline.predict_proba(sample)[0]
    classifier = pipeline.named_steps["classifier"]
    classes = list(classifier.classes_)
    malignant_probability = float(probabilities[classes.index(1)])

    return jsonify({
        "model": MODEL_PACKAGE.get("model_name", "Random Forest"),
        "prediction": "Malignant" if prediction == 1 else "Benign",
        "malignant_probability": malignant_probability,
        "message": "Hasil klasifikasi model untuk demonstrasi pembelajaran, bukan diagnosis medis."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
