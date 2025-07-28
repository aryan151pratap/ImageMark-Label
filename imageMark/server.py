from flask import Flask, request, jsonify
from flask_cors import CORS
from ultralytics import YOLO
import cv2, base64
import numpy as np
import re

app = Flask(__name__)

# Allow requests from your frontend
CORS(app, resources={r"/*": {"origins": ["http://192.168.27.50:5173", "http://localhost:5173"]}})

@app.after_request
def after_request(response):
    response.headers.add("Access-Control-Allow-Origin", "http://192.168.27.50:5173")
    response.headers.add("Access-Control-Allow-Headers", "Content-Type,Authorization")
    response.headers.add("Access-Control-Allow-Methods", "GET,POST,OPTIONS")
    return response

# Load YOLO model
model = YOLO("my_model.onnx")

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == "OPTIONS":
        return jsonify({"status": "ok"}), 200  # preflight quick response

    data = request.get_json()
    if not data or 'image' not in data:
        return jsonify({"error": "No image field in request"}), 400

    # Decode base64 image
    img_data = re.sub('^data:image/.+;base64,', '', data['image'])
    img_bytes = base64.b64decode(img_data)
    npimg = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Run YOLO prediction
    results = model(img)
    boxes = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            boxes.append({
                "x": int(x1),
                "y": int(y1),
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "label": label,
                "confidence": round(conf, 2)
            })

    return jsonify({"boxes": boxes})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
