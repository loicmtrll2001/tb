from ultralytics import YOLO
import cv2

# === CONFIGURATION ===
MODEL_PATH = "../runs/detect/ball_detector3/weights/best.pt"  # chemin du modèle entraîné
VIDEO_PATH = "../assets/test_videos_5c5/action_19.MOV"         # ta vidéo d'entrée

# === CHARGER LE MODÈLE ===
model = YOLO(MODEL_PATH)

# === LIRE LA VIDÉO ===
cap = cv2.VideoCapture(VIDEO_PATH)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Prédiction sur la frame
    results = model.predict(source=frame, conf=0.25, imgsz=640, verbose=False)[0]

    # Dessiner les boxes
    if results.boxes is not None:
        for box in results.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Ball", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Redimensionner juste pour l'affichage
    display_frame = cv2.resize(frame, (960, 540))

    cv2.imshow('Predictions', display_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
