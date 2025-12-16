from flask import Flask, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import base64
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(50))

with app.app_context():
    db.create_all()

@app.route('/recognize', methods=['POST'])
def recognize():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"result": "DENIED"})

    image_bytes = base64.b64decode(data['image'])
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({"result": "DENIED"})

    # TEMP LOGIC (replace with real face recognition)
    return jsonify({"result": "AUTHORIZED"})

if __name__ == "__main__":
    app.run(debug=True)
