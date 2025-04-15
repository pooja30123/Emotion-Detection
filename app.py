from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load trained model
model = load_model("models\model5.h5")
emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]
emotion_emojis = {
    "Angry": "😠",
    "Disgust": "😷",
    "Fear": "😨",
    "Happy": "😊",
    "Neutral": "😐",
    "Sad": "😢",
    "Surprise": "😲"
}

def preprocess_image(image_data):
    image_data = image_data.split(",")[1]  
    image_bytes = base64.b64decode(image_data)
    np_arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (48, 48))  
    img = img.reshape(1, 48, 48, 1) / 255.0  
    return img

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    img = preprocess_image(data["image"])
    pred = model.predict(img)
    emotion = emotion_labels[np.argmax(pred)]
    emoji = emotion_emojis.get(emotion, "😶")
    return jsonify({"emotion": emotion, "emoji": emoji})

if __name__ == "__main__":
    app.run(debug=True)
