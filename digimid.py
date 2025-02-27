# Import necessary libraries
from google.colab import files
import cv2
import numpy as np
import librosa
import time
from deepface import DeepFace
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns

# Function to analyze facial expressions
def analyze_face(frame):
    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
        emotion = result[0]['dominant_emotion']
        return emotion
    except Exception as e:
        print(f"Error analyzing face: {e}")
        return "Unknown"

# Function to analyze voice stress
def analyze_voice(audio, sr):
    pitch = np.mean(librosa.yin(audio, fmin=50, fmax=300))
    energy = np.mean(librosa.feature.rms(y=audio))

    # Define stress threshold
    stress_level = "High" if pitch > 200 and energy > 0.05 else "Low"
    return stress_level

# Function to determine if the person is lying based on trends
def detect_lie(emotion_list, stress_list):
    high_stress_count = stress_list.count("High")
    nervous_emotions = ["fear", "sad", "angry", "surprise"]
    nervous_count = sum(1 for e in emotion_list if e in nervous_emotions)

    if high_stress_count > 7 and nervous_count > 7:
        return " LIE DETECTED "
    elif high_stress_count > 5 or nervous_count > 5:
        return "Possibly Lying "
    else:
        return " Truth "

# Upload MP4 video file
video_file = "test.mp4" # Get the uploaded file name

# Store data for analysis
emotion_history = []
stress_history = []
nervous_counts = []
high_stress_counts = []
start_time = time.time()

print(" Lie Detection Started... Please wait for 1 minute.")

# Load the video file
cap = cv2.VideoCapture(video_file)

# Check if the video file was opened successfully
if not cap.isOpened():
    print(f"Error opening video file: {video_file}")
    exit()

# Get the frame rate of the video
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps * 2)  # Number of frames to skip for 2 seconds

frame_count = 0  # Initialize frame counter

# Process each frame from the video
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Analyze the frame every 2 seconds
    if frame_count % frame_interval == 0:
        # Analyze the frame
        emotion = analyze_face(frame)

        # Simulate audio analysis (you can replace this with actual audio extraction if needed)
        # For now, we will use a placeholder audio file
        audio_file = 'testaudio.mp3'  # Replace with your audio file path
        audio, sample_rate = librosa.load(audio_file, sr=None)  # Load audio file
        stress = analyze_voice(audio, sample_rate)

        # Store data for analysis
        emotion_history.append(emotion)
        stress_history.append(stress)

        # Count nervous emotions and high stress
        nervous_emotions = ["fear", "sad", "angry", "surprise"]
        nervous_count = 1 if emotion in nervous_emotions else 0
        high_stress_count = 1 if stress == "High" else 0

        nervous_counts.append(nervous_count)
        high_stress_counts.append(high_stress_count)
        print(nervous_counts)
        print(high_stress_counts)

        # Print real-time data in terminal
        print(f"Time: {frame_count / fps:.2f} seconds | Emotion: {emotion} | Stress Level: {stress}")

    frame_count += 1  # Increment frame counter

# Final analysis after processing
final_result = detect_lie(emotion_history, stress_history)

# Print final result in terminal
print(" FINAL RESULT: ", final_result, "\n")

# Perform linear regression only if there is data to process
if len(nervous_counts) > 0:
    X = np.array(range(len(nervous_counts))).reshape(-1, 1)  # Time steps
    y = np.array(nervous_counts)  # High stress counts
    model = LinearRegression()
    model.fit(X, y)

    y_pred = model.predict(X)

    # Plotting the results
    plt.figure(figsize=(10, 5))
    plt.scatter(X, y, color='blue', label='High Stress Count')
    plt.plot(X, y, color='blue', label='High Stress Count')
    plt.plot(X, y_pred, color='red', label='Linear Regression Line')
    plt.title('High Stress Count Over Time')
    plt.xlabel('Time (seconds)')
    plt.ylabel('High Stress Count')
