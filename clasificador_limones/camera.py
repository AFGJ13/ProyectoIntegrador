import cv2
import time
import serial
import threading
from ultralytics import YOLO

# ─── CONFIGURACIÓN ─────────────────────────────
PORT  = "COM6"
BAUD  = 9600
model = YOLO("best.pt")

cap = cv2.VideoCapture(1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ─── SERIAL ────────────────────────────────────
arduino = serial.Serial(PORT, BAUD, timeout=1)
time.sleep(2)

# ─── ESTADO COMPARTIDO ─────────────────────────
# Última detección del modelo (actualizada por el hilo de cámara)
ultima_deteccion = None
deteccion_lock   = threading.Lock()

# Flag: el Arduino pidió un valor
arduino_esperando = False
espera_lock       = threading.Lock()


def label_a_valor(label: str):
    """Convierte etiqueta YOLO a número 1/2/3."""
    if label == "desecho":
        return 1
    elif label == "exportacion":
        return 2
    elif label == "local":
        return 3
    return None


# ══════════════════════════════════════════════
# HILO: lee el serial del Arduino
# Cuando recibe "escanear", envía la última
# detección que tenga guardada.
# ══════════════════════════════════════════════
def leer_serial():
    global arduino_esperando

    while True:
        try:
            linea = arduino.readline().decode('utf-8', errors='ignore').strip()
            if not linea:
                continue

            print(f"Arduino → {linea}")

            if linea.lower() == "escanear":
                print("⚡ Arduino pidió decisión. Buscando detección...")

                # Esperar hasta 4 s a tener una detección
                deadline = time.time() + 4.0
                enviado  = False

                while time.time() < deadline:
                    with deteccion_lock:
                        det = ultima_deteccion

                    if det is not None:
                        valor = label_a_valor(det)
                        if valor is not None:
                            arduino.write(str(valor).encode())
                            print(f"✅ Enviado al Arduino: {valor} ({det})")
                            enviado = True
                            break

                    time.sleep(0.1)

                if not enviado:
                    # Sin detección clara → enviar 1 (desecho) como fallback
                    arduino.write(b'1')
                    print("⚠ Sin detección en 4 s → enviado fallback: 1")

        except Exception as e:
            print(f"Error serial: {e}")
            time.sleep(0.5)


# ══════════════════════════════════════════════
# MAIN — cámara + YOLO
# ══════════════════════════════════════════════
print("=" * 50)
print("   YOLO → ARDUINO (sincronizado)")
print("=" * 50)

threading.Thread(target=leer_serial, daemon=True).start()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results   = model(frame)
    annotated = results[0].plot()
    boxes     = results[0].boxes

    if boxes is not None and len(boxes) > 0:
        cls_id = int(boxes.cls[0])
        label  = model.names[cls_id]
        conf   = float(boxes.conf[0])

        if conf >= 0.5:          # solo guardar detecciones con buena confianza
            with deteccion_lock:
                ultima_deteccion = label
            print(f"Detectado: {label} | {conf:.2f}")
    else:
        # Si no hay nada en frame, limpiar
        with deteccion_lock:
            ultima_deteccion = None

    # Mostrar etiqueta actual en pantalla
    with deteccion_lock:
        etiqueta_actual = ultima_deteccion or "sin deteccion"

    cv2.putText(annotated, f"Ultimo: {etiqueta_actual}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("YOLO + ARDUINO", annotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
arduino.close()