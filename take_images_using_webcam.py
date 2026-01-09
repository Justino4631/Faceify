import cv2
import os
import time
from image_filters import crop_image, convert_to_grayscale

def take_screenshots(name:str):

    print("\nIn 5 seconds the camera will start and will take constant screenshots of you")
    print("Make sure to stay in frame and don't stay still: it's much better to get a wide variety of photos of you")
    print("For example, move your head around (not too much) and point it in all directions")
    print("You can smile, frown, move your eyebrows, etc...\n")
    print("The process will automatically end in 30 seconds")
    time.sleep(5)
    
    os.makedirs(f"screenshots/{name}", exist_ok=True)

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam")
        exit()

    interval = 30
    image_num = 0
    start_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        cv2.imshow("Webcam", frame)


        edited_frame = crop_image(convert_to_grayscale(frame))

        if edited_frame is not None:
            image_num += 1
            filename = f"screenshots/{name}/{name}{image_num}.jpg"
            cv2.imwrite(filename, edited_frame)
            time.sleep(0.03)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if time.time() - start_time >= interval or image_num >= 425:
            break

    cap.release()
    cv2.destroyAllWindows()

take_screenshots(name=input("Enter the name of the person who will be screenshotted (no spaces): "))