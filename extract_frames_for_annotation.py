import cv2
import os

# === CONFIGURATION ===
VIDEOS_DIR = "assets/test_videos_5c5"  # Dossier avec tes vidéos
OUTPUT_DIR = "dataset/frames"
FRAMES_PER_VIDEO = 10  # Nombre de frames extraites par vidéo
FRAME_START_OFFSET = 30  # Commencer à 30 frames pour éviter les débuts vides

# Créer l'output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Lister toutes les vidéos
videos = [v for v in os.listdir(VIDEOS_DIR) if v.endswith((".mp4", ".MOV", ".avi"))]

frame_count_total = 0

for video_name in videos:
    video_path = os.path.join(VIDEOS_DIR, video_name)
    cap = cv2.VideoCapture(video_path)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_interval = max((total_frames - FRAME_START_OFFSET) // FRAMES_PER_VIDEO, 1)

    frame_idx = FRAME_START_OFFSET
    extracted = 0

    while cap.isOpened() and extracted < FRAMES_PER_VIDEO:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        output_filename = f"{os.path.splitext(video_name)[0]}_frame{frame_idx}.jpg"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        cv2.imwrite(output_path, frame)

        frame_idx += frames_interval
        extracted += 1
        frame_count_total += 1

    cap.release()

print(f"✅ Extraction terminée : {frame_count_total} frames sauvegardées dans {OUTPUT_DIR}")
