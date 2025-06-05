from flask import Flask, request, jsonify
from PIL import Image
import numpy as np
import io
import tensorflow as tf
import os
import gdown
from flask_cors import CORS

from utils.preprocess import preprocess_for_detection, preprocess_for_segmentation
from utils.predict import predict_label, predict_mask

app = Flask(__name__)
CORS(app)

# === Google Drive Model IDs ===

# === Construct Download URLs ===
DETECTION_MODEL_URL = f"https://drive.google.com/uc?id={DETECTION_MODEL_ID}"
SEGMENTATION_MODEL_URL = f"https://drive.google.com/uc?id={SEGMENTATION_MODEL_ID}"

# === Local File Paths ===
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
DETECTION_MODEL_PATH = os.path.join(MODELS_DIR, "detection_model.keras")
SEGMENTATION_MODEL_PATH = os.path.join(MODELS_DIR, "segmentation_model.keras")

# === Download models if not present ===
def download_model(path, url, name):
    if not os.path.exists(path):
        print(f"🔽 Downloading {name} model...")
        gdown.download(url, path, quiet=False)
    else:
        print(f"✅ {name} model already exists.")

download_model(DETECTION_MODEL_PATH, DETECTION_MODEL_URL, "Detection")
download_model(SEGMENTATION_MODEL_PATH, SEGMENTATION_MODEL_URL, "Segmentation")

# === Load Models ===
try:
    print("⚙️ Loading detection model...")
    detection_model = tf.keras.models.load_model(DETECTION_MODEL_PATH)
    print("✅ Detection model loaded.")

    print("⚙️ Loading segmentation model...")
    segmentation_model = tf.keras.models.load_model(SEGMENTATION_MODEL_PATH)
    print("✅ Segmentation model loaded.")
except Exception as e:
    print("❌ Error loading models:", str(e))
    raise e

# === API Endpoint ===
@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    try:
        image_file = request.files['image']
        image = Image.open(io.BytesIO(image_file.read())).convert("RGB")

        # --- Tumor Detection ---
        detect_input = preprocess_for_detection(image)
        predicted_label, has_tumor = predict_label(detection_model, detect_input)

        response = {
            "label": predicted_label,
            "tumor_detected": has_tumor
        }

        # --- Tumor Segmentation ---
        if has_tumor:
            seg_input = preprocess_for_segmentation(image)
            mask = predict_mask(segmentation_model, seg_input)
            response["segmentation"] = mask.tolist()

        return jsonify(response)

    except Exception as e:
        print("❌ Prediction error:", str(e))
        return jsonify({"error": str(e)}), 500

# === Run App ===
if __name__ == "__main__":
    app.run(debug=True)
