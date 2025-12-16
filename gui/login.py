import cv2
import numpy as np
from PIL import Image
import sqlite3
import tkinter as tk
from tkinter import messagebox

face_cascade = cv2.CascadeClassifier("../backend/haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("../backend/model/trainer.yml")

conn = sqlite3.connect("../backend/database.db")
c = conn.cursor()

def recognize_face():
    cam = cv2.VideoCapture(0)
    recognized_user = None
    while True:
        ret, frame = cam.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
            if conf < 60:
                c.execute("SELECT name FROM user WHERE id=?", (id_,))
                row = c.fetchone()
                if row:
                    recognized_user = row[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"Authorized: {recognized_user}", (x, y-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(frame, "Unauthorized", (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.imshow("Face Login", frame)
        if cv2.waitKey(1) & 0xFF == ord('q') or recognized_user:
            break
    cam.release()
    cv2.destroyAllWindows()
    return recognized_user

def login():
    user = recognize_face()
    if user:
        c.execute("INSERT INTO access_log (user_name, status) VALUES (?, ?)", (user, "AUTHORIZED"))
        conn.commit()
        messagebox.showinfo("Login Successful", f"Welcome, {user}!")
    else:
        c.execute("INSERT INTO access_log (user_name, status) VALUES (?, ?)", (None, "UNAUTHORIZED"))
        conn.commit()
        messagebox.showerror("Login Failed", "Access Denied")

root = tk.Tk()
root.title("Face Login System")
root.geometry("300x150")

tk.Label(root, text="Press Login and show your face").pack(pady=20)
tk.Button(root, text="Login", command=login).pack(pady=10)

root.mainloop()
