import cv2
import numpy as np
import os
import csv
from datetime import datetime
from collections import deque

# -----------------------------------
# CONFIG
# -----------------------------------
CASCADE_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
MODEL_FILE = "trainer.yml"
LABEL_MAP_FILE = "label_map.npy"
ATTENDANCE_FILE = "attendance.csv"

FACE_SIZE_MIN = 130
CONFIDENCE_THRESHOLD = 45
FRAMES_REQUIRED = 8   # SAME FACE IN 8 FRAMES

# -----------------------------------
# LOAD MODELS
# -----------------------------------
face_cascade = cv2.CascadeClassifier(CASCADE_FILE)
if face_cascade.empty():
    raise IOError("Haarcascade not loaded")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read(MODEL_FILE)

label_map = np.load(LABEL_MAP_FILE, allow_pickle=True).item()

# -----------------------------------
# ATTENDANCE SETUP
# -----------------------------------
if not os.path.exists(ATTENDANCE_FILE):
    with open(ATTENDANCE_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["StudentID", "Date", "Time"])

marked_students = set()
recent_predictions = deque(maxlen=FRAMES_REQUIRED)

# -----------------------------------
# CAMERA
# -----------------------------------
cap = cv2.VideoCapture(0)
print("📷 Camera started. Hold face steady...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=6,
        minSize=(FACE_SIZE_MIN, FACE_SIZE_MIN)
    )

    if len(faces) != 1:
        recent_predictions.clear()
        cv2.imshow("Attendance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    (x, y, w, h) = faces[0]
    roi_gray = gray[y:y+h, x:x+w]
    roi_gray = cv2.resize(roi_gray, (200, 200))

    label_id, confidence = recognizer.predict(roi_gray)

    if confidence < CONFIDENCE_THRESHOLD and label_id in label_map:
        student_id = label_map[label_id]
        recent_predictions.append(student_id)
        label = f"{student_id} ({int(confidence)})"
        color = (0, 255, 0)
    else:
        recent_predictions.clear()
        label = "Unknown"
        color = (0, 0, 255)

    # -------------------------------
    # FINAL VERIFICATION
    # -------------------------------
    if len(recent_predictions) == FRAMES_REQUIRED:
        if len(set(recent_predictions)) == 1:
            confirmed_id = recent_predictions[0]

            if confirmed_id not in marked_students:
                now = datetime.now()
                with open(ATTENDANCE_FILE, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        confirmed_id,
                        now.strftime("%Y-%m-%d"),
                        now.strftime("%H:%M:%S")
                    ])
                marked_students.add(confirmed_id)
                print(f"✅ Attendance CONFIRMED for {confirmed_id}")

            recent_predictions.clear()

    cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    cv2.putText(frame, label, (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Program ended")
