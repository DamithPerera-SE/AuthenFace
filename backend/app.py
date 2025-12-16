from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import base64
import cv2
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Database config
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# =========================
# MODELS
# =========================
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)

class AccessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_name = db.Column(db.String(50))
    status = db.Column(db.String(50))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

# Create tables
with app.app_context():
    db.create_all()

# =========================
# FACE MODEL
# =========================
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists("model/trainer.yml"):
    recognizer.read("model/trainer.yml")

# =========================
# ROUTES
# =========================
@app.route("/")
def home():
    return "AuthenFace Backend Running"

@app.route("/recognize", methods=["POST"])
def recognize():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"result": "DENIED"})

        image_bytes = base64.b64decode(data["image"])
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        result = "No Face Detected"
        name = None

        for (x, y, w, h) in faces:
            try:
                user_id, conf = recognizer.predict(gray[y:y+h, x:x+w])
                if conf < 60:
                    user = User.query.get(user_id)
                    if user:
                        result = "AUTHORIZED"
                        name = user.name
                else:
                    result = "UNAUTHORIZED"
            except:
                result = "MODEL NOT TRAINED"

        # Log access attempt
        log = AccessLog(user_name=name, status=result)
        db.session.add(log)
        db.session.commit()

        return jsonify({"result": result, "user": name})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
