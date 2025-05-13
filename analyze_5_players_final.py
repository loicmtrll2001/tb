import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from tensorflow.keras.models import load_model
import mediapipe as mp
import joblib
import os
import sys

sys.path.append("ai_gestures")
from ai_gestures.feature_extraction import extract_features_from_keypoints

# config
VIDEO_PATH = "../assets/test_videos_5c5/action_14.MOV"
MODEL_GESTES_PATH = "../models/model_gestes.h5"
LABEL_ENCODER_PATH = "../models/label_encoder.pkl"
YOLO_BALL_MODEL = "../runs/detect/ball_detector3/weights/best.pt"
YOLO_PLAYER_MODEL = "yolov8n.pt"

#chargement des modèles
ball_model = YOLO(YOLO_BALL_MODEL)
player_model = YOLO(YOLO_PLAYER_MODEL)
tracker = DeepSort(max_age=300)
gesture_model = load_model(MODEL_GESTES_PATH)
label_encoder = joblib.load(LABEL_ENCODER_PATH)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1)

cap = cv2.VideoCapture(VIDEO_PATH)
player_sequences = {i: [] for i in range(1, 6)}  # IDs 1 à 5
last_ball_holder = None
passes_detected = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))
    original = frame.copy()

    #détection des joueurs
    player_dets = player_model.predict(source=frame, conf=0.3, classes=[0], verbose=False)[0]
    boxes = player_dets.boxes.xyxy.cpu().numpy() if player_dets.boxes is not None else []
    confidences = player_dets.boxes.conf.cpu().numpy() if player_dets.boxes is not None else []

    detections = []
    for box, conf in zip(boxes, confidences):
        x1, y1, x2, y2 = box
        detections.append(((float(x1), float(y1), float(x2 - x1), float(y2 - y1)), float(conf)))

    tracks = tracker.update_tracks(detections, frame=frame)
    tracked_players = {}

    for track in tracks:
        if not track.is_confirmed():
            continue
        track_id = int(track.track_id)
        if not (1 <= track_id <= 5):
            continue

        x1, y1, x2, y2 = map(int, track.to_ltrb())
        tracked_players[track_id] = (x1, y1, x2, y2)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        #prédiction des gestes
        cropped = original[y1:y2, x1:x2]
        if cropped.size == 0:
            continue
        result = pose.process(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))

        if result.pose_landmarks:
            keypoints = [(lm.x, lm.y, lm.z) for lm in result.pose_landmarks.landmark]
            features = extract_features_from_keypoints(keypoints)
            player_sequences[track_id].append(features)

            if len(player_sequences[track_id]) >= 18:
                seq = np.array(player_sequences[track_id][-18:]).flatten()
                if seq.shape[0] < 540:
                    seq = np.pad(seq, (0, 540 - seq.shape[0]))
                seq = seq.reshape(1, 540)
                pred = gesture_model.predict(seq, verbose=0)
                gesture = label_encoder.inverse_transform([np.argmax(pred)])[0]
                cv2.putText(frame, gesture, (x1, y2 + 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 50, 50), 2)


    #detecte la balle et l'attribue à un joueur
    current_ball_holder = None
    ball_results = ball_model.predict(source=original, conf=0.3, imgsz=640, verbose=False)[0]
    ball_centers = []

    for box in ball_results.boxes.xyxy.cpu().numpy() if ball_results.boxes is not None else []:
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        ball_centers.append((cx, cy))
        cv2.circle(frame, (cx, cy), 8, (0, 255, 0), -1)

    if ball_centers:
        bx, by = ball_centers[0]
        min_dist = float('inf')
        for pid, (x1, y1, x2, y2) in tracked_players.items():
            px, py = (x1 + x2) // 2, (y1 + y2) // 2
            dist = np.linalg.norm(np.array([bx, by]) - np.array([px, py]))
            if dist < min_dist and dist < 100:
                min_dist = dist
                current_ball_holder = pid

        if current_ball_holder:
            cv2.putText(frame, f"Balle: Joueur {current_ball_holder}",
                        (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 200, 255), 3)

    #detection de la passe
    if current_ball_holder is not None:
        if current_ball_holder != last_ball_holder and last_ball_holder is not None:
            print(f"Passe : Joueur {last_ball_holder} -> Joueur {current_ball_holder}")
            passes_detected.append((last_ball_holder, current_ball_holder))
        last_ball_holder = current_ball_holder

    cv2.imshow("Analyse 5 Joueurs", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
