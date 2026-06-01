from ultralytics import YOLO

# Cargar modelo pequeño (ideal para tu caso)
model = YOLO("yolov8n.pt")

# Entrenamiento
model.train(
    data="dataset/data.yaml",
    epochs=80,
    imgsz=320,
    batch=8,
    device="cpu",  # si no tienes GPU
    name="limones_model"
)