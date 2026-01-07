import cv2
import os
import numpy as np
#  configuration 
FACES_FOLDER = "faces"
MODEL_FILE = "trainer.yml"
IMAGE_SIZE = (200, 200)   
# Recognizer
recognizer = cv2.face.LBPHFaceRecognizer_create()

faces = []
labels = []
label_map = {}
current_label = 0
# For loading face data
for student_id in os.listdir(FACES_FOLDER):
    student_path = os.path.join(FACES_FOLDER, student_id)

    if not os.path.isdir(student_path):
        continue

    current_label += 1
    label_map[current_label] = student_id

    for img_name in os.listdir(student_path):
        if img_name.lower().endswith(".jpg"):
            img_path = os.path.join(student_path, img_name)

            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                print(f" Cannot read {img_path}")
                continue

            img = cv2.resize(img, IMAGE_SIZE)

            faces.append(img)
            labels.append(current_label)

if len(faces) == 0:
    print(" No faces found for training")
    exit()

#For training model
recognizer.train(faces, np.array(labels))
recognizer.save(MODEL_FILE)

np.save("label_map.npy", label_map)

print(" Training completed successfully")
print(" Students trained:")
for k, v in label_map.items():
    print(f"  Label {k} -> Student {v}")
