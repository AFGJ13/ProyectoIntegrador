import os
os.environ["OMP_NUM_THREADS"] = "4"

import cv2
import numpy as np
import onnxruntime as ort
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
# MODELO ONNX
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

print("✅ Modelo ONNX listo")

# ==========================
# PREPROCESS
# ==========================

def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ==========================
# POSTPROCESS
# ==========================

def postprocess(frame, outputs):

    predictions = outputs[0][0].T
    h, w, _ = frame.shape

    boxes = []
    scores = []
    class_ids = []

    for pred in predictions:

        conf = pred[4]
        if conf < CONF_THRESHOLD:
            continue

        class_scores = pred[5:]
        class_id = np.argmax(class_scores)
        score = class_scores[class_id]

        if score < CONF_THRESHOLD:
            continue

        cx, cy, bw, bh = pred[:4]

        x1 = int((cx - bw / 2) * w / INPUT_SIZE)
        y1 = int((cy - bh / 2) * h / INPUT_SIZE)

        bw = int(bw * w / INPUT_SIZE)
        bh = int(bh * h / INPUT_SIZE)

        boxes.append([x1, y1, bw, bh])
        scores.append(float(score))
        class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD)

    if len(indices) > 0:
        for i in indices.flatten():

            x, y, bw, bh = boxes[i]
            class_id = class_ids[i]
            score = scores[i]

            label = f"{CLASSES[class_id]} {score:.2f}"

            cv2.rectangle(frame, (x,y), (x+bw,y+bh), (0,255,0), 2)
            cv2.putText(frame, label, (x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0,255,0), 2)

    return frame

# ==========================
# CAMARA
# ==========================

cap = cv2.VideoCapture(0)
cap.set(3, 320)
cap.set(4, 240)

frame_count = 0
cached = None
prev_time = time.time()

# ==========================
# LOOP
# ==========================

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # 🔥 SOLO IA CADA 3 FRAMES
    if frame_count % 3 == 0:

        input_tensor = preprocess(frame)

        outputs = session.run(
            None,
            {input_name: input_tensor}
        )

        cached = postprocess(frame.copy(), outputs)

    display = cached if cached is not None else frame

    # FPS
    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cv2.putText(display, f"FPS: {int(fps)}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (255,255,255), 2)

    cv2.imshow("YOLO Raspberry", display)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()