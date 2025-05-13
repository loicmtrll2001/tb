import os
import cv2
import mediapipe as mp
import csv
from tqdm import tqdm
from feature_extraction import extract_features_from_keypoints

# === CONFIGURATION ===
DATA_DIR = "../assets/training_videos"
OUTPUT_CSV = "../models/dataset_sequences.csv"
FRAMES_PER_SEQUENCE = 30

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# === Initialiser l'en-tête du CSV dynamiquement ===
sample_features = extract_features_from_keypoints([[0, 0, 0, 0]] * 33)
header = [f"f{i}" for i in range(len(sample_features) * FRAMES_PER_SEQUENCE)]
header.append("label")

def normalize_label(label_name):
    label = label_name.lower()
    if label in ["dribble_sur_place", "dribble_en_deplacement", "dribble"]:
        return "dribble"
    elif label == "none":
        return "none"
    elif label == "layup":
        return "layup"
    elif label == "shoot":
        return "shoot"
    return label

def extract_sequences_from_video(video_path, label):
    cap = cv2.VideoCapture(video_path)
    frames = []
    sequences = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            keypoints = []
            for landmark in results.pose_landmarks.landmark:
                keypoints.append([landmark.x, landmark.y, landmark.z, landmark.visibility])

            features = extract_features_from_keypoints(keypoints)
            frames.append(features)

            if len(frames) == FRAMES_PER_SEQUENCE:
                sequence = []
                for f in frames:
                    sequence.extend(f)
                sequence.append(label)
                sequences.append(sequence)
                frames = []

    cap.release()
    return sequences

def main():
    all_sequences = []

    print(f"Extraction depuis : {DATA_DIR}")
    for folder in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, folder)
        if not os.path.isdir(folder_path):
            continue

        label = normalize_label(folder)

        print(f"Traitement des vidéos pour le label : {label}")
        for filename in tqdm(os.listdir(folder_path)):
            if not filename.endswith((".mp4", ".mov", ".avi")):
                continue

            video_path = os.path.join(folder_path, filename)
            sequences = extract_sequences_from_video(video_path, label)
            all_sequences.extend(sequences)

    print(f"Écriture du fichier : {OUTPUT_CSV}")
    with open(OUTPUT_CSV, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(all_sequences)

    print("Extraction terminée")

if __name__ == "__main__":
    main()