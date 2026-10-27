import cv2
import os

def capture_face(name):

    cam = cv2.VideoCapture(0)

    if not cam.isOpened():
        print("Camera not working")
        return

    print("Press 's' to capture image")

    while True:
        ret, frame = cam.read()

        if not ret:
            break

        cv2.imshow("Capture Face", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            file_path = f"face_data/{name}.jpg"
            cv2.imwrite(file_path, frame)
            print("Image saved:", file_path)
            break

        elif key == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

