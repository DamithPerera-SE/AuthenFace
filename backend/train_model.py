import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

faces = []
ids = []

for user_folder in os.listdir("dataset"):
    if not user_folder.startswith("User."):
        continue

    user_id = int(user_folder.split(".")[1])
    folder_path = os.path.join("dataset", user_folder)

    for image_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image_name)
        pil_img = Image.open(img_path).convert("L")
        img_np = np.array(pil_img, "uint8")
        detected_faces = detector.detectMultiScale(img_np)

        for (x,y,w,h) in detected_faces:
            faces.append(img_np[y:y+h, x:x+w])
            ids.append(user_id)

os.makedirs("model", exist_ok=True)
recognizer.train(faces, np.array(ids))
recognizer.save("model/trainer.yml")
print("Model trained successfully")
