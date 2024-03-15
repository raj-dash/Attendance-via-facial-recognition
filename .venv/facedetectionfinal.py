import cv2
import numpy as np
import os
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

recognizer = cv2.face.LBPHFaceRecognizer_create()
capture_count_file = "capture_count.txt"
latest_captured_filename = None

def load_capture_count():
    if os.path.exists(capture_count_file):
        with open(capture_count_file, 'r') as file:
            return int(file.read())
    else:
        return 0

def save_capture_count(count):
    with open(capture_count_file, 'w') as file:
        file.write(str(count))

def load_recognizer():
    if os.path.exists("facedata.yml") and os.path.getsize("facedata.yml") > 0:
        recognizer.read("facedata.yml")
        print("Face recognition model loaded successfully.")
    else:
        print("No pre-trained data found. Capture and store faces first.")

def save_recognizer():
    try:
        recognizer.save("facedata.yml")
        print("Face recognition model saved successfully.")
    except cv2.error as e:
        print(f"Error saving facedata.yml: {e}")

def capture_and_store_face():
    global capture_count, latest_captured_filename

    cap = cv2.VideoCapture(0)
    start_time = time.time()
    face_captured = False

    while cap.isOpened() and not face_captured:
        ret, frame = cap.read()
        if not ret:
            break

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.resize(roi_gray, (100, 100))
                    filename = f"captured_face_{capture_count}.jpg"  # Use a unique filename
                    cv2.imwrite(filename, roi_gray)
                    print("Face captured and stored successfully as:", filename)
                    latest_captured_filename = filename
                    face_captured = True
                    cap.release()
                    break

        if time.time() - start_time > 3:
            print("Timeout: Face capture unsuccessful.")
            cap.release()
            break

    capture_count += 1

def recognize_faces():
    load_recognizer()
    face_recognized = False

    cap = cv2.VideoCapture(0)
    start_time = time.time()

    while cap.isOpened() and not face_recognized:
        ret, frame = cap.read()
        if not ret:
            break

        with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    bboxC = detection.location_data.relative_bounding_box
                    ih, iw, _ = frame.shape
                    x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
                    roi_gray = cv2.cvtColor(frame[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY)
                    roi_gray = cv2.resize(roi_gray, (100, 100))

                    label, confidence = recognizer.predict(roi_gray)

                    if confidence < 70:
                        print("Person recognized. Registration Number:", label)
                        face_recognized = True
                    else:
                        pass

        if time.time() - start_time > 3:
            print("Recognition failed. Unknown person.")
            cap.release()
            break

if __name__ == "__main__":
    capture_count = load_capture_count()

    while True:
        key = input("Press 'r' to recognize, 's' to capture and store, or 'q' to quit: ")

        if key == 'q':
            save_capture_count(capture_count)
            break

        elif key == 'r':
            recognize_faces()

        elif key == 's':
            capture_and_store_face()

            if latest_captured_filename:
                label = input(f"Enter Registration Number for {latest_captured_filename}: ")
                if label:
                    images, labels = [], []
                    captured_face = cv2.imread(latest_captured_filename, cv2.IMREAD_GRAYSCALE)
                    images.append(captured_face)
                    labels.append(int(label))

                    recognizer.update(np.asarray(images), np.asarray(labels, dtype=np.int32))
                    save_recognizer()
                else:
                    print("No registration number entered. The captured image will not be stored.")

        else:
            print("Invalid input. Please try again.")
