import cv2
import numpy as np
import onnxruntime as ort

from config import MODEL_PATH, IMAGE_SIZE, CONFIDENCE_THRESHOLD


class Detector:

    def __init__(self):

        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=["CPUExecutionProvider"]
        )

        self.input_name = self.session.get_inputs()[0].name

        with open("labels.txt", "r") as f:

            self.labels = [line.strip() for line in f.readlines()]

    def preprocess(self, frame):

        image = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = image.astype(np.float32) / 255.0

        image = np.transpose(image, (2, 0, 1))

        image = np.expand_dims(image, axis=0)

        return image

    def predict(self, frame):

        input_tensor = self.preprocess(frame)

        outputs = self.session.run(
            None,
            {self.input_name: input_tensor}
        )

        predictions = outputs[0]

        predictions = np.squeeze(predictions)

        if len(predictions.shape) == 2:

            predictions = predictions.T

        best_class = None
        best_confidence = 0

        for detection in predictions:

            class_scores = detection[4:]

            class_id = np.argmax(class_scores)

            confidence = class_scores[class_id]

            if confidence > best_confidence:

                best_confidence = confidence
                best_class = class_id

        if best_confidence < CONFIDENCE_THRESHOLD:

            return "desecho"

        return self.labels[best_class]
