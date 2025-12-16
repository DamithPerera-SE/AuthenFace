import cv2
import os
from app import db, User

cam = cv2.VideoCapture(0)
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name = input("Enter User Name: ")
user = User(name=name)
db.session.add(user)
db.session.commit()
user_id = user.id
print(f"Assigned User ID: {user_id}")

count = 0
os.makedirs(f"dataset/User.{user_id}", exist_ok=True)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"dataset/User.{user_id}/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0),2)

    cv2.imshow("Capturing Faces", img)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()
print(f"Captured {count} images for {name}")
