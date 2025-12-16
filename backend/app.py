from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
import base64
import os
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# =========================
# DATABASE MODELS
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)

class AccessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(100))
    status = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# ✅ FIX: CREATE DB INSIDE APP CONTEXT
with app.app_context():
    db.create_all()

# =========================
# LOAD FACE MODELS
# =========================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()

if os.path.exists("model/trainer.yml"):
    recognizer.read("model/trainer.yml")

# =========================
# FACE RECOGNITION API
# =========================
@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.json
    img_data = base64.b64decode(data["image"])
    np_arr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    result = "No Face Detected"
    name = None

    for (x, y, w, h) in faces:
        try:
            user_id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            if confidence < 60:
                user = User.query.get(user_id)
                if user:
                    result = "AUTHORIZED"
                    name = user.name
            else:
                result = "UNAUTHORIZED"
        except:
            result = "MODEL NOT TRAINED"

    log = AccessLog(user_name=name, status=result)
    db.session.add(log)
    db.session.commit()

    return jsonify({"result": result, "user": name})

if __name__ == "__main__":
    app.run(debug=True)
