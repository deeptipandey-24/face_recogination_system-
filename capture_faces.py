import cv2
import os

FACES_FOLDER = "faces"
IMAGES_PER_STUDENT = 20
CAMERA_ID = 0
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

if face_cascade.empty():
    print("❌ Haar Cascade NOT loaded")
    exit()
# Input student ID
student_id = input("Enter Student ID (100, 101, etc): ").strip()
student_folder = os.path.join(FACES_FOLDER, student_id)
os.makedirs(student_folder, exist_ok=True)
# Camera Start
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    print("❌ Camera not accessible")
    exit()

count = 0
print("📷 Camera started")
print("➡ Look straight at the camera")
print("➡ Press Q to quit")

# Capture Loop

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5,
        minSize=(100, 100)
    )

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]

        if count < IMAGES_PER_STUDENT:
            img_path = os.path.join(
                student_folder,
                f"img{count + 1}.jpg"
            )
            cv2.imwrite(img_path, face_img)
            print(f"✅ Saved {img_path}")
            count += 1

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.putText(
        frame,
        f"Images: {count}/{IMAGES_PER_STUDENT}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imshow("Capture Faces", frame)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= IMAGES_PER_STUDENT:
        break
cap.release()
cv2.destroyAllWindows()

print("face capture completed")
