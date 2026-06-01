import cv2

for i in range(5):  # prueba 0 a 4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Camara detectada en ID: {i}")
        cap.release()