from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

# ---------------- MODEL ----------------
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))

with app.app_context():
    db.create_all()

# ---------------- ROUTES ----------------
@app.route("/")
def home():
    return "AuthenFace Backend Running"

@app.route("/recognize", methods=["POST"])
def recognize():
    data = request.get_json()

    if not data or "image" not in data:
        return jsonify({"result": "DENIED"})

    try:
        image_bytes = base64.b64decode(data["image"])
        np_img = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"result": "DENIED"})

        # TEMP AUTH LOGIC
        return jsonify({"result": "AUTHORIZED"})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ---------------- RUN ----------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
