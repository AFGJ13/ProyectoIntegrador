from ultralytics import YOLO

model = YOLO("runs/detect/limones_model/weights/best.pt")

model.predict(
    source=1,
    imgsz=320,
    conf=0.4
)