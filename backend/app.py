from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
import cv2
import numpy as np
from PIL import Image
import os
import base64
from datetime import datetime

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

# Database models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50), nullable=False)

class AccessLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, nullable=True)
    user_name = db.Column(db.String(50))
    status = db.Column(db.String(20))
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

db.create_all()

# Load Haar cascade
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Load trained LBPH model
recognizer = cv2.face.LBPHFaceRecognizer_create()
if os.path.exists("model/trainer.yml"):
    recognizer.read("model/trainer.yml")

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    img_data = base64.b64decode(data['image'])
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    result = "No Face Detected"
    user_id = None
    user_name = None

    for (x, y, w, h) in faces:
        id_, conf = recognizer.predict(gray[y:y+h, x:x+w])
        if conf < 60:
            user = User.query.filter_by(id=id_).first()
            if user:
                result = f"Authorized: {user.name}"
                user_id = user.id
                user_name = user.name
        else:
            result = "Unauthorized Face"

    # Log access attempt
    log = AccessLog(user_id=user_id, user_name=user_name, status=result)
    db.session.add(log)
    db.session.commit()

    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(debug=True)
