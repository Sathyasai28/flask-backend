import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import google.generativeai as genai
import logging

# 🔧 Add pyngrok to expose Flask server to the internet
# from pyngrok import ngrok

app = Flask(__name__)
CORS(app)
logging.basicConfig(level=logging.INFO)

# # 🔧 Start ngrok tunnel
# public_url = ngrok.connect(5000)
# logging.info(f"🔗 Public URL: {public_url}")

GEMINI_API_KEY = "AIzaSyCoiINCfXVqr3Ndl9PcFuZ37ZOfgd-KYno"
genai.configure(api_key=GEMINI_API_KEY)

# Load model
model = tf.keras.models.load_model("trained_plant_disease_model.keras")
logging.info("✅ Model loaded successfully!")

CLASS_NAMES = [  # Same list as before — no changes here
    "Apple__Apple_scab", "Apple_Black_rot", "Apple_Cedar_apple_rust", "Apple__healthy",
    "Blueberry__healthy", "Cherry_Powdery_mildew", "Cherry_healthy",
    "Corn_Cercospora_leaf_spot", "Corn_Common_rust", "Corn_Northern_Leaf_Blight",
    "Corn_healthy", "Grape_Black_rot", "Grape_Esca", "Grape_Leaf_blight", "Grape_healthy",
    "Orange_Citrus_greening", "Peach_Bacterial_spot", "Peach_healthy",
    "Pepper_Bacterial_spot", "Pepper_healthy", "Potato_Early_blight",
    "Potato_Late_blight", "Potato_healthy", "Raspberry_healthy", "Soybean_healthy",
    "Squash_Powdery_mildew", "Strawberry_Leaf_scorch", "Strawberry_healthy",
    "Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight",
    "Tomato_Leaf_Mold", "Tomato_Septoria_leaf_spot", "Tomato_Spider_mites",
    "Tomato_Target_Spot", "Tomato_Yellow_Leaf_Curl_Virus", "Tomato_Tomato_mosaic_virus",
    "Tomato_healthy"
]

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        query = data.get('query', '')
        disease = data.get('disease', '')

        if not query or not disease:
            return jsonify({"error": "Missing query or disease"}), 400

        prompt = f"""
        The user has a plant disease identified as "{disease}". 
        They are asking: "{query}"
        Provide helpful expert info including:
        - Description
        - Causes
        - Symptoms
        - Treatments
        - Prevention
        """

        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(prompt)

        return jsonify({"response": response.text})
    except Exception as e:
        logging.error(f"Error during chat: {e}")
        return jsonify({"error": str(e)}), 500

def preprocess_image(image):
    try:
        image = Image.open(io.BytesIO(image)).convert('RGB')
        image = image.resize((128, 128))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array
    except Exception as e:
        logging.error(f"Image preprocessing error: {str(e)}")
        raise

@app.route("/", methods=["GET"])
def home():
    return "🌿 Welcome to the Crop Disease Prediction API! Use /predict to get predictions."

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_data = file.read()
    logging.info("📩 Image received for prediction.")

    try:
        processed_image = preprocess_image(image_data)
        predictions = model.predict(processed_image)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        predicted_class_name = CLASS_NAMES[predicted_class]

        top_3_indices = np.argsort(predictions[0])[-3:][::-1]
        top_3_predictions = [
            {'class_name': CLASS_NAMES[i], 'confidence': f"{predictions[0][i]*100:.2f}%"}
            for i in top_3_indices
        ]

        logging.info(f"✅ Prediction: {predicted_class_name} with confidence {confidence*100:.2f}%")
        return jsonify({
            'prediction': predicted_class_name,
            'confidence': f"{confidence*100:.2f}%",
            'top_3_predictions': top_3_predictions
        })

    except Exception as e:
        logging.error(f"❌ Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=10000)
