from ultralytics import YOLO

model = YOLO("runs/detect/limones_model/weights/best.pt")

model.export(
    format="onnx",
    imgsz=256,        # 🔥 más rápido para Pi
    simplify=True,
    opset=12
)