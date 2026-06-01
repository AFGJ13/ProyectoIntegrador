import time

from camera import Camera
from detector import Detector
from classifier import Classifier

from config import DETECTION_TIME


cam = Camera()

detector = Detector()

classifier = Classifier()

print("Sistema iniciado")


while True:

    input("\nPresiona ENTER para iniciar deteccion...")

    predictions = []

    start_time = time.time()

    while time.time() - start_time < DETECTION_TIME:

        frame = cam.read()

        if frame is None:
            continue

        result = detector.predict(frame)

        predictions.append(result)

        print("Prediccion:", result)

    final_result = classifier.average_prediction(predictions)

    print("\n===================")
    print("RESULTADO FINAL:")
    print(final_result)
    print("===================")
