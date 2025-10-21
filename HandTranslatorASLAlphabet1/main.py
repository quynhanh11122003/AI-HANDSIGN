# import os
# import cv2                              # Xử lý hình ảnh & video
# import torch                            # Framework deep learning (dùng cho model ResNet)
# import mediapipe as mp                  # Phát hiện tay theo thời gian thực
# from torchvision import transforms
# from PIL import Image                   # Xử lý ảnh với PIL
# import time

# # ─── Các thiết lập môi trường để loại bỏ các cảnh báo nội bộ ─────────────
# # Đặt biến môi trường để TF Lite không dùng feedback tensors (nếu được hỗ trợ)
# os.environ["MEDIAPIPE_DISABLE_INFERENCE_FEEDBACK"] = "1"

# # Nếu cài absl-py, có thể thiết lập log level của absl xuống ERROR
# try:
#     from absl import logging as absl_logging
#     absl_logging.set_verbosity(absl_logging.ERROR)
# except ImportError:
#     pass

# # ─── Fix SSL chứng chỉ để tải dữ liệu nếu cần ───────────────────────────────
# import ssl, certifi
# ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# # ─── Import model (ResNet, ASLDataset) đã được viết theo API mới ───────────
# # Đảm bảo file model.py đã thay đổi sử dụng:
# #    resnet18(weights=ResNet18_Weights.DEFAULT)
# from model import ResNet, ASLDataset

# # ─── Kết nối camera: Ưu tiên webcam máy tính, nếu không mở được thì thử IP cam ─────────
# cap = cv2.VideoCapture(0)  # Ưu tiên webcam máy tính
# if not cap.isOpened():
#     print("[!] Không mở được webcam mặc định, thử kết nối IP cam...")
#     IP_CAM_URL = 'http://192.168.0.102:8080/video'
#     cap = cv2.VideoCapture(IP_CAM_URL)
#     if not cap.isOpened():
#         print(f"[!] Không mở được IP cam tại {IP_CAM_URL}. Thoát.")
#         exit(1)

# # ─── Khởi tạo mô hình AI và các tham số
# num_classes = 29  # Số lượng ký tự ASL + các ký tự đặc biệt
# model = ResNet(num_classes)
# model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
# model.eval()

# # Tiền xử lý ảnh
# preprocess = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # ─── Khởi tạo Mediapipe Hands với cấu hình mới ─────────────────────────────
# # Chế độ static_image_mode=True sẽ chạy phát hiện trên mỗi khung hình, đảm bảo chuẩn hình ảnh (và loại bỏ cảnh báo về IMAGE_DIMENSIONS)
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     static_image_mode=False,        # Chạy detection trên mỗi frame
#     max_num_hands=2,
#     model_complexity=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.5
# )

# # ─── Hàm chung lấy ROI của tay dựa vào loại tay ("Left" hoặc "Right") ───────
# def track_hand(frame, results, hand_type="left"):
#     frame_height, frame_width, _ = frame.shape
#     if not results.multi_handedness:
#         return None, None, None
#     hand_index = None
#     # Duyệt qua các kết quả để tìm tay có label tương ứng
#     for idx, hand_handedness in enumerate(results.multi_handedness):
#         if hand_handedness.classification[0].label == hand_type:
#             hand_index = idx
#             break
#     if hand_index is None:
#         return None, None, None
#     hand_landmarks = results.multi_hand_landmarks[hand_index]
#     hand_center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame_width)
#     hand_center_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame_height)
#     roi_size = 250  # ROI đảm bảo hình vuông
#     top_left_x = max(0, hand_center_x - roi_size // 2)
#     top_left_y = max(0, hand_center_y - roi_size // 2)
#     bottom_right_x = min(frame_width, hand_center_x + roi_size // 2)
#     bottom_right_y = min(frame_height, hand_center_y + roi_size // 2)
#     roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
#     return roi, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

# def track_left_hand(frame, results):
#     return track_hand(frame, results, hand_type="Left")

# def track_right_hand(frame, results):
#     return track_hand(frame, results, hand_type="Right")

# # ─── Biến để chọn chế độ theo dõi: False -> tay trái, True -> tay phải ───────
# use_right_hand = True  # Đổi thành True nếu muốn theo dõi tay phải

# # ─── Khởi tạo các biến trạng thái ─────────────────────────────
# flip_frame = False
# sentence = ""                   # Chuỗi văn bản kết quả
# prev_label = None               # Nhãn ký hiệu trước đó
# last_prediction_time = 0        # Thời gian nhận dạng ký tự cuối cùng
# prediction_delay = 5.0          # Nhận 1 ký tự mỗi 5 giây
# recording = False               # Trạng thái ghi văn bản

# # ─── Từ điển ánh xạ ký hiệu ASL ─────────────────────────────
# gestos = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
#     6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
#     12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
#     18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
#     24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
# }

# # ─── Vòng lặp chính xử lý hình ảnh từ camera ─────────────────────────────
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Không nhận được khung hình. Kiểm tra kết nối.")
#         break

#     if flip_frame:
#         frame = cv2.flip(frame, 1)

#     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     current_time = time.time()

#     if results.multi_hand_landmarks:
#         # Chọn tay theo biến use_right_hand
#         if use_right_hand:
#             roi, top_left, bottom_right = track_right_hand(frame, results)
#         else:
#             roi, top_left, bottom_right = track_left_hand(frame, results)
        
#         if roi is not None:
#             roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#             input_tensor = preprocess(roi_pil).unsqueeze(0)

#             with torch.no_grad():
#                 output = model(input_tensor)
#             pred = torch.argmax(output, dim=1).item()
#             label = gestos.get(pred, 'Unknown')

#             if recording:
#                 if label != prev_label and (current_time - last_prediction_time) > prediction_delay:
#                     if label == 'space':
#                         sentence += " "
#                     elif label == 'del':
#                         sentence = sentence[:-1]
#                     elif label != 'nothing':
#                         sentence += label

#                     prev_label = label
#                     last_prediction_time = current_time
#                     print("📄 Văn bản hiện tại:", sentence)

#             mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
#             cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
#             cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (255, 255, 255), 2)

#     cv2.putText(frame, sentence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.2, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.imshow("Hand Tracking", frame)
    
#     key = cv2.waitKey(1) & 0xFF

#     if key == ord('b'):          # Bắt đầu ghi
#         recording = True
#         print(" BẮT ĐẦU GHI")
#     elif key == ord('d'):        # Tạm dừng ghi
#         recording = False
#         print("⏸ TẠM DỪNG")
#     elif key == ord('r'):        # Reset văn bản
#         sentence = ""
#         prev_label = None
#         print(" Reset câu")
#     elif key == ord(' '):        # Thêm dấu cách
#         sentence += " "
#         print(" Thêm khoảng trắng:", sentence)
#     elif key in (8, 127):        # Xóa ký tự cuối (Backspace)
#         sentence = sentence[:-1]
#         print(" Xóa ký tự, câu hiện tại:", sentence)
#     elif key == 27:              # Thoát
#         break

# cap.release()
# cv2.destroyAllWindows()

# import os
# import cv2
# import torch
# import mediapipe as mp
# import numpy as np
# from torchvision import transforms
# from PIL import Image
# import time
# import ssl, certifi

# # Tắt cảnh báo nội bộ của MediaPipe
# os.environ["MEDIAPIPE_DISABLE_INFERENCE_FEEDBACK"] = "1"
# try:
#     from absl import logging as absl_logging
#     absl_logging.set_verbosity(absl_logging.ERROR)
# except ImportError:
#     pass
# ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# # Import model
# from model import ResNet, ASLDataset

# # Camera
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("[!] Không mở được webcam, thử IP cam...")
#     IP_CAM_URL = 'http://192.168.0.102:8080/video'
#     cap = cv2.VideoCapture(IP_CAM_URL)
#     if not cap.isOpened():
#         print(f"[!] Không mở IP cam tại {IP_CAM_URL}")
#         exit(1)

# # Model AI
# num_classes = 29
# model = ResNet(num_classes)
# model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
# model.eval()

# # Tiền xử lý ảnh
# preprocess = transforms.Compose([
#     transforms.Resize((128, 128)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                          std=[0.229, 0.224, 0.225]),
# ])

# # MediaPipe
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
#                        model_complexity=1, min_detection_confidence=0.7,
#                        min_tracking_confidence=0.5)

# mp_selfie_segmentation = mp.solutions.selfie_segmentation
# selfie_segmentation = mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

# # Hàm tracking
# def track_hand(frame, results, hand_type="Left"):
#     frame_height, frame_width, _ = frame.shape
#     if not results.multi_handedness:
#         return None, None, None
#     hand_index = None
#     for idx, hand_handedness in enumerate(results.multi_handedness):
#         if hand_handedness.classification[0].label == hand_type:
#             hand_index = idx
#             break
#     if hand_index is None:
#         return None, None, None
#     hand_landmarks = results.multi_hand_landmarks[hand_index]
#     hand_center_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * frame_width)
#     hand_center_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * frame_height)
#     roi_size = 250
#     top_left_x = max(0, hand_center_x - roi_size // 2)
#     top_left_y = max(0, hand_center_y - roi_size // 2)
#     bottom_right_x = min(frame_width, hand_center_x + roi_size // 2)
#     bottom_right_y = min(frame_height, hand_center_y + roi_size // 2)
#     roi = frame[top_left_y:bottom_right_y, top_left_x:bottom_right_x]
#     return roi, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y)

# track_left_hand = lambda frame, results: track_hand(frame, results, "Left")
# track_right_hand = lambda frame, results: track_hand(frame, results, "Right")

# # Trạng thái
# use_right_hand = True
# flip_frame = False
# sentence = ""
# prev_label = None
# last_prediction_time = 0
# prediction_delay = 5.0
# recording = False

# gestos = {
#     0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
#     6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
#     12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
#     18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
#     24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'
# }

# # Main loop
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("Không nhận được khung hình.")
#         break

#     if flip_frame:
#         frame = cv2.flip(frame, 1)

#     # Làm mờ nền
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results_seg = selfie_segmentation.process(frame_rgb)
#     condition = results_seg.segmentation_mask > 0.5
#     blurred_frame = cv2.GaussianBlur(frame, (55, 55), 0)
#     frame = np.where(condition[..., None], frame, blurred_frame)

#     results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
#     current_time = time.time()

#     if results.multi_hand_landmarks:
#         if use_right_hand:
#             roi, top_left, bottom_right = track_right_hand(frame, results)
#         else:
#             roi, top_left, bottom_right = track_left_hand(frame, results)

#         if roi is not None:
#             roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
#             input_tensor = preprocess(roi_pil).unsqueeze(0)

#             with torch.no_grad():
#                 output = model(input_tensor)
#             pred = torch.argmax(output, dim=1).item()
#             label = gestos.get(pred, 'Unknown')

#             if recording:
#                 if label != prev_label and (current_time - last_prediction_time) > prediction_delay:
#                     if label == 'space':
#                         sentence += " "
#                     elif label == 'del':
#                         sentence = sentence[:-1]
#                     elif label != 'nothing':
#                         sentence += label
#                     prev_label = label
#                     last_prediction_time = current_time
#                     print("📄 Văn bản hiện tại:", sentence)

#             mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
#             cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
#             cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
#                         1, (255, 255, 255), 2)

#     cv2.putText(frame, sentence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
#                 1.2, (255, 255, 255), 2, cv2.LINE_AA)
#     cv2.imshow("Hand Tracking", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('b'):
#         recording = True
#         print("BẮT ĐẦU GHI")
#     elif key == ord('d'):
#         recording = False
#         print("⏸ TẠM DỪNG")
#     elif key == ord('r'):
#         sentence = ""
#         prev_label = None
#         print("Reset câu")
#     elif key == ord(' '):
#         sentence += " "
#         print("Thêm khoảng trắng:", sentence)
#     elif key in (8, 127):
#         sentence = sentence[:-1]
#         print("Xóa ký tự:", sentence)
#     elif key == 27:
#         break

# cap.release()
# cv2.destroyAllWindows()


import os
import cv2
import torch
import mediapipe as mp
from torchvision import transforms
from PIL import Image
import numpy as np
import time
import ssl, certifi

# Fix SSL để tải model nếu cần
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

# Tắt cảnh báo MediaPipe
os.environ["MEDIAPIPE_DISABLE_INFERENCE_FEEDBACK"] = "1"
try:
    from absl import logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except ImportError:
    pass

from model import ResNet, ASLDataset

# Kết nối camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    IP_CAM_URL = 'http://192.168.0.102:8080/video'
    cap = cv2.VideoCapture(IP_CAM_URL)
    if not cap.isOpened():
        print("[!] Không mở được camera.")
        exit(1)

# Load mô hình
num_classes = 29
model = ResNet(num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# MediaPipe setup với độ nhạy cao hơn
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    model_complexity=1,
    min_detection_confidence=0.75,   # tăng độ nhạy
    min_tracking_confidence=0.7      # tăng giữ tay
)
selfie_segmentation = mp.solutions.selfie_segmentation.SelfieSegmentation(model_selection=1)

def track_hand(frame, results, hand_type="Left"):
    h, w, _ = frame.shape
    if not results.multi_handedness:
        return None, None, None
    hand_index = None
    for idx, handedness in enumerate(results.multi_handedness):
        if handedness.classification[0].label == hand_type:
            hand_index = idx
            break
    if hand_index is None:
        return None, None, None
    landmarks = results.multi_hand_landmarks[hand_index]
    cx = int(landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].x * w)
    cy = int(landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP].y * h)
    roi_size = 350
    x1 = max(0, cx - roi_size // 2)
    y1 = max(0, cy - roi_size // 2)
    x2 = min(w, cx + roi_size // 2)
    y2 = min(h, cy + roi_size // 2)
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1), (x2, y2)

track_left_hand = lambda frame, results: track_hand(frame, results, "Left")
track_right_hand = lambda frame, results: track_hand(frame, results, "Right")

# Cấu hình
use_right_hand = True
flip_frame = False
sentence = ""
prev_label = None
last_prediction_time = 0
prediction_delay = 5
recording = False

gestos = {i: c for i, c in enumerate(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ['del', 'nothing', 'space'])}

# Vòng lặp chính
while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình")
        break

    if flip_frame:
        frame = cv2.flip(frame, 1)

    # Xóa nền bằng làm mờ
    fg_mask = selfie_segmentation.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).segmentation_mask
    condition = fg_mask > 0.5
    blurred = cv2.GaussianBlur(frame, (55, 55), 0)
    frame = np.where(condition[..., None], frame, blurred)

    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    current_time = time.time()

    if results.multi_hand_landmarks:
        roi, top_left, bottom_right = track_right_hand(frame, results) if use_right_hand else track_left_hand(frame, results)
        if roi is not None:
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            input_tensor = preprocess(roi_pil).unsqueeze(0)
            with torch.no_grad():
                output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            label = gestos.get(pred, 'Unknown')

            if recording and label != prev_label and (current_time - last_prediction_time) > prediction_delay:
                if label == 'space':
                    sentence += " "
                elif label == 'del':
                    sentence = sentence[:-1]
                elif label != 'nothing':
                    sentence += label
                prev_label = label
                last_prediction_time = current_time
                print("📄 Văn bản hiện tại:", sentence)

            mp_drawing.draw_landmarks(frame, results.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

            # Hiển thị nhãn có viền đen
            cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(frame, f"Label: {label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2, cv2.LINE_AA)

    # Hiển thị văn bản có viền đen
    cv2.putText(frame, sentence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, sentence, (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                1.2, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow("Hand Tracking", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        recording = True
        print(" BẮT ĐẦU GHI")
    elif key == ord('d'):
        recording = False
        print("⏸ TẠM DỪNG")
    elif key == ord('r'):
        sentence = ""
        prev_label = None
        print(" Reset câu")
    elif key == ord(' '):
        sentence += " "
        print(" Thêm khoảng trắng:", sentence)
    elif key in (8, 127):
        sentence = sentence[:-1]
        print(" Xóa ký tự, câu hiện tại:", sentence)
    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()




