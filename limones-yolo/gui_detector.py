import cv2
import time
from ultralytics import YOLO

# =====================================
# CONFIGURACIÓN
# =====================================
MODEL_PATH = "runs/detect/limones_model/weights/best.pt"
CAMERA_ID = 1   # 👈 Webcam externa
IMG_SIZE = 320  # Resolución para inferencia (rápido)

# =====================================
# CARGAR MODELO
# =====================================
model = YOLO(MODEL_PATH)

# =====================================
# INICIAR CÁMARA
# =====================================
cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    print("❌ Error: no se pudo abrir la cámara")
    exit()

# Resolución de captura (puedes bajar a 480x360 si va lento)
cap.set(3, 640)  # ancho
cap.set(4, 480)  # alto

print("✅ Cámara iniciada correctamente")

# =====================================
# LOOP PRINCIPAL
# =====================================
while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("❌ Error al leer frame")
        break

    # =====================================
    # INFERENCIA YOLO
    # =====================================
    results = model(frame, imgsz=IMG_SIZE, conf=0.4)

    # =====================================
    # DIBUJAR RESULTADOS
    # =====================================
    annotated_frame = results[0].plot()

    # =====================================
    # CALCULAR FPS
    # =====================================
    end_time = time.time()
    fps = 1 / (end_time - start_time)

    # =====================================
    # TEXTO EN PANTALLA
    # =====================================
    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        annotated_frame,
        "Detector de Limones Tahiti",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2
    )

    # =====================================
    # MOSTRAR VENTANA
    # =====================================
    cv2.imshow("Vision Artificial - Limones", annotated_frame)

    # =====================================
    # SALIR CON ESC
    # =====================================
    if cv2.waitKey(1) & 0xFF == 27:
        print("🛑 Saliendo...")
        break

# =====================================
# LIBERAR RECURSOS
# =====================================
cap.release()
cv2.destroyAllWindows()