import cv2
import numpy as np
import os
import mediapipe as mp
import time

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

recognizer = cv2.face.LBPHFaceRecognizer_create()

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
                    cv2.imwrite("captured_face.jpg", roi_gray)
                    print("Face captured and stored successfully.")
                    face_captured = True
                    cap.release()
                    break

        if time.time() - start_time > 3:
            print("Timeout: Face capture unsuccessful.")
            cap.release()
            break

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

while True:
    key = input("Press 'r' to recognize, 's' to capture and store, or 'q' to quit: ")

    if key == 'q':
        break

    elif key == 'r':
        recognize_faces()

    elif key == 's':
        capture_and_store_face()

        images, labels = [], []
        captured_face = cv2.imread("captured_face.jpg", cv2.IMREAD_GRAYSCALE)
        images.append(captured_face)
        labels.append(input("Enter Registration Number: "))

        recognizer.update(np.asarray(images), np.asarray(labels, dtype=np.int32))
        save_recognizer()

    else:
        print("Invalid input. Please try again.")