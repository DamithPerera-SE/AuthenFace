import cv2
import os
from app import app, db, User

detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)

name = input("Enter User Name: ")

# ✅ FIX: APP CONTEXT
with app.app_context():
    user = User(name=name)
    db.session.add(user)
    db.session.commit()
    user_id = user.id

print(f"User ID: {user_id}")

path = f"dataset/User.{user_id}"
os.makedirs(path, exist_ok=True)

count = 0

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        count += 1
        cv2.imwrite(f"{path}/{count}.jpg", gray[y:y+h, x:x+w])
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("Capture Faces", img)

    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
        break

cam.release()
cv2.destroyAllWindows()

print("Face capture complete")
