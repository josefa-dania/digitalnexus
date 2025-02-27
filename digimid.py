# Install necessary libraries
!pip install flask flask-cors pyngrok librosa opencv-python-headless deepface numpy

# Import dependencies
import os
import cv2
import numpy as np
import librosa
import time
from flask import Flask, request, jsonify
from pyngrok import ngrok
from deepface import DeepFace
app = Flask(__name__)


# Allow requests from frontend
from flask_cors import CORS
CORS(app)

# Function to analyze facial expressions
def analyze_face(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except Exception:
        return "Unknown"

# Function to analyze voice stress
def analyze_voice(audio, sr):
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    energy = np.mean(librosa.feature.rms(y=audio))
    return "High" if pitch > 200 and energy > 0.05 else "Low"

# API route to process uploaded video
@app.route("/upload", methods=["POST"])
def upload():
    file = request.files['video']
    filename = "uploaded_video.mp4"
    file.save(filename)

    cap = cv2.VideoCapture(filename)
    emotion_history, stress_history = [], []
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Analyze frame
        emotion = analyze_face(frame)
        stress = "Low"  # Placeholder (replace with real audio analysis)

        emotion_history.append(emotion)
        stress_history.append(stress)

    cap.release()
    
    result = {"message": "Processing complete", "emotions": emotion_history, "stress": stress_history}
    return jsonify(result)

# Start Flask & ngrok
port = 5000
ngrok.set_auth_token("2tdHE19Tj55xgPKXkCUOmlXspeK_6LWi9MWoqq6tXxHbbz1ik")
public_url = ngrok.connect(port).public_url
print(f"Public URL: {public_url}")

app.run(port=port)