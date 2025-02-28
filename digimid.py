from flask import Flask, render_template, request
import cv2
import numpy as np
import librosa
import os
import time
from deepface import DeepFace

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Function to analyze facial expressions
def analyze_face(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        return result[0]['dominant_emotion']
    except:
        return "Unknown"

# Function to analyze voice stress
def analyze_voice(audio, sr):
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    energy = np.mean(librosa.feature.rms(y=audio))
    return "High" if pitch > 200 and energy > 0.05 else "Low"

# Function to determine if the person is lying
def detect_lie(emotion_list, stress_list):
    high_stress_count = stress_list.count("High")
    nervous_emotions = ["fear", "sad", "angry", "surprise"]
    nervous_count = sum(1 for e in emotion_list if e in nervous_emotions)

    if high_stress_count > 7 and nervous_count > 7:
        return "LIE DETECTED"
    elif high_stress_count > 5 or nervous_count > 5:
        return "Possibly Lying"
    else:
        return "Truth"

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        video = request.files["video"]
        audio = request.files["audio"]

        video_path = os.path.join(app.config["UPLOAD_FOLDER"], video.filename)
        audio_path = os.path.join(app.config["UPLOAD_FOLDER"], audio.filename)

        video.save(video_path)
        audio.save(audio_path)

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * 2)

        emotion_history = []
        stress_history = []

        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                emotion = analyze_face(frame)
                audio_data, sample_rate = librosa.load(audio_path, sr=None)
                stress = analyze_voice(audio_data, sample_rate)

                emotion_history.append(emotion)
                stress_history.append(stress)

            frame_count += 1

        cap.release()
        result = detect_lie(emotion_history, stress_history)

        return render_template("index.html", result=result)

    return render_template("index.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
