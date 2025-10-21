from flask import Flask, render_template, request, jsonify
from PIL import Image
import torch
from torchvision import transforms
import cv2
import os
import base64
import uuid
import numpy as np
from io import BytesIO
import mediapipe as mp
from model import ResNet

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
num_classes = 29
model = ResNet(num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

gestos = {i: c for i, c in enumerate(
    list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space'])}

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('image')
    if not file:
        return render_template("index.html", result="Không có ảnh")

    filepath = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4().hex}.jpg")
    file.save(filepath)

    img = Image.open(filepath).convert("RGB")
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = gestos[pred]

    return render_template("index.html", image_name=os.path.basename(filepath), result=label)

@app.route('/predict_frame', methods=['POST'])
def predict_frame():
    data = request.json
    if 'image' not in data:
        return jsonify({'label': 'No image'})

    # Decode base64
    img_data = data['image'].split(',')[1]
    img_bytes = base64.b64decode(img_data)
    np_img = cv2.imdecode(
        np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)

    # Detect hand với Mediapipe
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                        min_detection_confidence=0.7) as hands:
        results = hands.process(cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            h, w, _ = np_img.shape

            x_list = [lm.x for lm in hand_landmarks.landmark]
            y_list = [lm.y for lm in hand_landmarks.landmark]
            x1 = int(min(x_list) * w) - 20
            y1 = int(min(y_list) * h) - 20
            x2 = int(max(x_list) * w) + 20
            y2 = int(max(y_list) * h) + 20
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            roi = np_img[y1:y2, x1:x2]

            if roi.shape[0] == 0 or roi.shape[1] == 0:
                return jsonify({'label': 'Invalid ROI'})

            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)).resize((128, 128))
        else:
            return jsonify({'label': 'nothing'})

    input_tensor = transform(roi_pil).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        label = gestos[pred]

    return jsonify({'label': label})

if __name__ == '__main__':
    app.run(debug=True)
