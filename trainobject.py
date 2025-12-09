from ultralytics import YOLO

# 1. Load base model (small, fast)
model = YOLO("yolov8n.pt")

# 2. Train on your dataset
model.train(
    data="D:\PPE_KIT_DATASET\data.yaml",
    epochs=80,          # increase if needed
    imgsz=640,
    batch=8,
    workers=2,
    name="safetyshoe_detector"
)
