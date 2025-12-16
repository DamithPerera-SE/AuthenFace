import cv2
import numpy as np
from PIL import Image
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

path = "dataset"
face_samples = []
ids = []

for user_folder in os.listdir(path):
    folder_path = os.path.join(path, user_folder)
    if not os.path.isdir(folder_path):
        continue
    user_id = int(user_folder.split(".")[1])
    for image_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, image_name)
        img = Image.open(img_path).convert('L')
        img_numpy = np.array(img,'uint8')
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            face_samples.append(img_numpy[y:y+h,x:x+w])
            ids.append(user_id)

recognizer.train(face_samples, np.array(ids))
os.makedirs("model", exist_ok=True)
recognizer.save("model/trainer.yml")
print("[INFO] Training complete!")
