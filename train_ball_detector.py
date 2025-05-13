from ultralytics import YOLO


DATASET_YAML = "dataset/ball_dataset.yaml"


model = YOLO('yolov8n.pt')

model.train(
    data=DATASET_YAML,
    epochs=50,
    imgsz=640,
    batch=8,
    name="ball_detector",
    project="runs/detect",
    device='cpu'
)
