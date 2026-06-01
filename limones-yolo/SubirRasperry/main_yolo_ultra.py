import os
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import numpy as np
import onnxruntime as ort
import threading
import time

# ==========================
# CONFIG
# ==========================

MODEL_PATH = "best.onnx"
INPUT_SIZE = 256

CONF_THRESHOLD = 0.4
NMS_THRESHOLD = 0.45

CLASSES = ["desecho", "local", "exportacion"]

# ==========================
# MODELO
# ==========================

so = ort.SessionOptions()
so.intra_op_num_threads = 4
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(
    MODEL_PATH,
    sess_options=so,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name

print("✅ YOLO ONNX listo")

# ==========================
# VARIABLES GLOBALES
# ==========================

frame = None
processed_frame = None
lock = threading.Lock()

# ==========================
# PREPROCESS
# ==========================

def preprocess(img):
    img = cv2.resize(img, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ==========================
# POSTPROCESS OPTIMIZADO
# ==========================

def postprocess(frame, outputs):

    preds = outputs[0][0].T
    h, w, _ = frame.shape

    boxes = []
    scores = []
    class_ids = []

    for pred in preds:

        conf = pred[4]
        if conf < CONF_THRESHOLD:
            continue

        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score < CONF_THRESHOLD:
            continue

        cx, cy, bw, bh = pred[:4]

        x = int((cx - bw/2) * w / INPUT_SIZE)
        y = int((cy - bh/2) * h / INPUT_SIZE)
        bw = int(bw * w / INPUT_SIZE)
        bh = int(bh * h / INPUT_SIZE)

        boxes.append([x, y, bw, bh])
        scores.append(float(score))
        class_ids.append(class_id)

    # 🔥 LIMITAR DETECCIONES (MEGA IMPORTANTE)
    if len(boxes) > 5:
        boxes = boxes[:5]
        scores = scores[:5]
        class_ids = class_ids[:5]

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():

            x, y, bw, bh = boxes[i]
            cls = class_ids[i]
            score = scores[i]

            label = f"{CLASSES[cls]} {score:.2f}"

            cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0), 2)

    return frame

# ==========================
# THREAD IA
# ==========================

def ai_thread():
    global frame, processed_frame

    frame_count = 0

    while True:

        if frame is None:
            continue

        with lock:
            f = frame.copy()

        frame_count += 1

        # 🔥 SOLO IA CADA 4 FRAMES
        if frame_count % 4 != 0:
            continue

        input_tensor = preprocess(f)

        outputs = session.run(None, {input_name: input_tensor})

        result = postprocess(f, outputs)

        with lock:
            processed_frame = result

# ==========================
# THREAD CAMARA
# ==========================

def camera_thread():
    global frame

    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

    while True:
        ret, f = cap.read()
        if not ret:
            continue

        with lock:
            frame = f

# ==========================
# MAIN
# ==========================

def main():

    global processed_frame

    threading.Thread(target=camera_thread, daemon=True).start()
    threading.Thread(target=ai_thread, daemon=True).start()

    prev_time = time.time()

    while True:

        if processed_frame is None:
            continue

        with lock:
            display = processed_frame.copy()

        # FPS
        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        cv2.putText(display, f"FPS: {int(fps)}",
                    (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255,255,255), 2)

        cv2.imshow("YOLO ULTRA Raspberry", display)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()

# ==========================
# RUN
# ==========================

if __name__ == "__main__":
    main()