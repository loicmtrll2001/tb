import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
from feature_extraction import extract_features_from_keypoints
from collections import Counter, deque

VIDEO_PATH = "../assets/test_videos/shoot_11.mp4"  # change ici pour ta vidéo de test
MODEL_PATH = "../models/model_gestes.h5"
ENCODER_PATH = "../models/label_encoder.pkl"
FRAMES_PER_SEQUENCE = 30
VOTE_HISTORY_LENGTH = 5

print("Chargement du modèle et du label encoder")
model = load_model(MODEL_PATH)
encoder = joblib.load(ENCODER_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(VIDEO_PATH)
frames = []
predictions = []
vote_history = deque(maxlen=VOTE_HISTORY_LENGTH)

print("Lecture de la vidéo")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (960, 480))
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        keypoints = [[lm.x, lm.y, lm.z, lm.visibility] for lm in results.pose_landmarks.landmark]
        features = extract_features_from_keypoints(keypoints)
        frames.append(features)

        if len(frames) == FRAMES_PER_SEQUENCE:
            sequence = []
            for f in frames:
                sequence.extend(f)
            X_input = np.array(sequence).reshape(1, -1)

            pred = model.predict(X_input, verbose=0)

            class_id = np.argmax(pred)
            class_label = encoder.inverse_transform([class_id])[0]

            vote_history.append(class_label)

            most_common = Counter(vote_history).most_common(1)[0][0]

            if most_common == "none":
                print("Aucun geste détecté")
            else:
                print(f"Prédiction glissante : {most_common}")
            predictions.append(most_common)
            frames = []

cap.release()

print("Prédictions terminées")
if predictions:
    final_vote = Counter(predictions).most_common(1)[0]
    print(f"Prédiction finale (majoritaire) : {final_vote[0]} ({final_vote[1]} votes)")
else:
    print("Pas de prédiction possible (pas assez de frames ?)")
