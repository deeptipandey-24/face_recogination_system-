import os

# Absolute path to your faces folder
FACES_FOLDER = "/Users/deeptipandey/Desktop/UniAttend/faces"

for student_id in os.listdir(FACES_FOLDER):
    student_path = os.path.join(FACES_FOLDER, student_id)
    if os.path.isdir(student_path):
        print("Student folder found:", student_id)
        for img_name in os.listdir(student_path):
            if img_name.lower().endswith(".jpg"):
                print("  Image found:", img_name)
