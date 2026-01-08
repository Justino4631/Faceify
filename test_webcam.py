import cv2
import pyttsx3
from evaluate_model import predict_face
import os
from datetime import datetime
import time
import threading

def speak(text):
    engine = pyttsx3.init()

    engine.setProperty('rate', 180)
    engine.setProperty('volume', 1.0)

    voices = engine.getProperty('voices')
    engine.setProperty('voice', voices[1].id)
    engine.say(text)
    engine.runAndWait()
    
os.makedirs("screenshots", exist_ok=True)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
    exit()

interval = 5
last_capture_time = time.time()
image_num = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("Webcam", frame)

    current_time = time.time()

    if current_time - last_capture_time >= interval:
        last_capture_time = current_time

        image_num += 1
        filename = f"screenshots/Screenshot{image_num}.jpg"
        cv2.imwrite(filename, frame)
        face_prediction = predict_face(f"{filename}")

        threading.Thread(
            target=speak,
            args=(f"{face_prediction} detected",),
            daemon=True
        ).start()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()