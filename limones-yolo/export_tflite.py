from ultralytics import YOLO

model = YOLO("runs/detect/limones_model/weights/best.pt")

model.export(
    format="tflite",
    imgsz=320
)

print("✅ Exportación completada")