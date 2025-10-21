import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.models import resnet18, ResNet18_Weights
import os
from PIL import Image
from tqdm import tqdm
import mediapipe as mp
import numpy as np
import cv2


# Tải dataset từ KaggleHub (optional)
# path = kagglehub.dataset_download("grassknoted/asl-alphabet")
# print("Path to dataset files:", path)



# Khởi tạo MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

def extract_hand_landmarks(image):
    # Chuyển BGR sang RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Phát hiện landmark
    results = hands.process(image_rgb)

    # Khởi tạo mảng numpy để lưu tọa độ landmark
    landmarks_array = np.zeros((21, 3))  # 21 điểm landmark, mỗi điểm có x,y,z

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]  # Lấy landmark của bàn tay đầu tiên

        # Chuyển đổi landmark thành mảng numpy
        for idx, landmark in enumerate(hand_landmarks.landmark):
            landmarks_array[idx] = [landmark.x, landmark.y, landmark.z]

    return landmarks_array

# Dataset tùy chỉnh cho ASL
class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        categories = sorted(os.listdir(data_dir))
        for i, category in enumerate(categories):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                for file_name in os.listdir(category_path):
                    file_path = os.path.join(category_path, file_name)
                    if os.path.isfile(file_path):
                        self.file_list.append((file_path, i))
                self.class_to_idx[category] = i
                self.idx_to_class[i] = category

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        # Đọc ảnh bằng OpenCV
        image = cv2.imread(img_path)

        # Trích xuất landmark
        landmarks = extract_hand_landmarks(image)

        # Chuyển ảnh sang PIL để áp dụng transform
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        if self.transform:
            image = self.transform(image)

        # Ghép landmarks với tensor ảnh
        landmarks_tensor = torch.FloatTensor(landmarks.flatten())  # Làm phẳng mảng landmark
        return image, landmarks_tensor, label

# Định nghĩa mô hình ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # Tải ResNet18 pre-trained từ torchvision
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features

        # Thêm layer xử lý landmark
        self.landmark_fc = nn.Sequential(
            nn.Linear(63, 128),  # 21 landmark points * 3 (x,y,z)
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        # Layer kết hợp features từ ảnh và landmark
        self.fc = nn.Linear(num_ftrs + 128, num_classes)

        # Xóa fc layer cuối của ResNet
        self.resnet.fc = nn.Identity()

    def forward(self, image, landmarks):
        # Xử lý ảnh qua ResNet
        img_features = self.resnet(image)

        # Xử lý landmark
        landmark_features = self.landmark_fc(landmarks)

        # Kết hợp features
        combined = torch.cat((img_features, landmark_features), dim=1)

        # Đưa qua layer cuối
        output = self.fc(combined)
        return output

# Đường dẫn tới thư mục chứa dữ liệu (đã tải dataset từ Kaggle về)
data_dir = r"C:\Users\HIULT\OneDrive\Desktop\aPhong\HandTranslatorASLAlphabet1\archive\asl_alphabet_train\asl_alphabet_train"

# Tiền xử lý ảnh: resize, đảo ảnh ngẫu nhiên và chuyển sang tensor
# Tiền xử lý ảnh: resize, đảo ảnh ngẫu nhiên và chuyển sang tensor
transform = transforms.Compose([
    # Giữ nguyên các bước tiền xử lý ban đầu của bạn
    transforms.Grayscale(num_output_channels=3),  # Chuyển ảnh sang ảnh xám nhưng tạo ra 3 kênh
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(p=0.5),      # Lật ảnh ngẫu nhiên theo chiều ngang (tốt cho huấn luyện cả tay trái/phải)

    # --- Bắt đầu thêm các kỹ thuật Data Augmentation ---

    # 1. Xoay ảnh ngẫu nhiên:
    # Xoay ảnh một góc ngẫu nhiên trong khoảng từ -20 đến +20 độ.
    # Điều này giúp mô hình nhận diện được cử chỉ tay ở các góc nghiêng khác nhau.
    transforms.RandomRotation(degrees=20),

    # 2. Biến đổi Affine ngẫu nhiên:
    # Bao gồm dịch chuyển (translate) và co giãn (scale) nhẹ.
    #   degrees=0: không xoay thêm ở bước này (vì đã có RandomRotation).
    #   translate=(0.1, 0.1): dịch chuyển ảnh tối đa 10% theo chiều rộng và chiều cao.
    #   scale=(0.85, 1.15): co giãn kích thước ảnh ngẫu nhiên từ 85% đến 115% kích thước ban đầu.
    # Giúp mô hình robust hơn với các thay đổi nhỏ về vị trí và kích thước của tay trong khung hình.
    transforms.RandomAffine(degrees=0,
                            translate=(0.1, 0.1),
                            scale=(0.85, 1.15)),

    # 3. (Tùy chọn) Thay đổi màu sắc ngẫu nhiên (Color Jitter):
    #   brightness=0.3: thay đổi độ sáng ngẫu nhiên.
    #   contrast=0.3: thay đổi độ tương phản ngẫu nhiên.
    #   saturation=0.3: thay đổi độ bão hòa màu ngẫu nhiên.
    #   hue=0.1: thay đổi màu sắc (hue) ngẫu nhiên.
    # LƯU Ý QUAN TRỌNG: Vì bạn áp dụng `transforms.Grayscale` ở đầu,
    # `ColorJitter` sẽ có ít hoặc không có tác dụng vì ảnh đã được chuyển thành ảnh xám.
    # Nếu bạn muốn `ColorJitter` hiệu quả, nó nên được áp dụng *trước* khi ảnh được chuyển sang grayscale,
    # tức là áp dụng trên ảnh RGB gốc. Ví dụ:
    #
    # transforms.Compose([
    #     transforms.Resize((128, 128)),
    #     transforms.RandomHorizontalFlip(p=0.5),
    #     transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # Áp dụng trên ảnh màu
    #     transforms.RandomRotation(degrees=20),
    #     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.85, 1.15)),
    #     transforms.Grayscale(num_output_channels=3), # Sau đó mới Grayscale nếu vẫn muốn
    #     transforms.ToTensor(),
    #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # ])
    #
    # Hiện tại, tôi sẽ comment dòng ColorJitter này lại vì nó không hiệu quả với cấu trúc Grayscale hiện tại.
    # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),

    # --- Kết thúc thêm Data Augmentations ---

    transforms.ToTensor(),                       # Chuyển ảnh PIL thành Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], # Chuẩn hóa Tensor
                         std=[0.229, 0.224, 0.225]),
])



# Tạo dataset và lấy số lớp dựa vào thư mục
dataset = ASLDataset(data_dir, transform=transform)
# Lấy danh sách các categories từ dataset (nếu cần):
categories = os.listdir(data_dir)
categories.sort()
print("Số lượng mục:", len(categories), flush=True)
num_classes = len(dataset.class_to_idx)

# Chia dataset thành train (70%), validation (15%) và test (15%)
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# Tạo DataLoader cho các tập dữ liệu
batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Khởi tạo mô hình, loss function và optimizer
model = ResNet(num_classes)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Thiết bị: ưu tiên GPU nếu khả dụng
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Huấn luyện
num_epochs = 2
best_val_loss = float('inf')

print("Bắt đầu huấn luyện...")
try:
    # Cập nhật training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}") as pbar:
            for images, landmarks, labels in train_loader:
                images = images.to(device)
                landmarks = landmarks.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images, landmarks)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                pbar.update(1)
                pbar.set_postfix(loss=f'{train_loss/pbar.n:.4f}')

        # Sau mỗi epoch, đánh giá trên tập validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, landmarks, labels in val_loader:
                images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
                outputs = model(images, landmarks)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_loss /= len(val_loader)
        accuracy = 100 * correct / total

        print(f"Epoch [{epoch + 1}/{num_epochs}] - Loss: {val_loss:.4f}, Validation Accuracy: {accuracy:.2f}%")

        # Lưu mô hình nếu loss trên validation được cải thiện
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("Saved best_model.pth.")

        # (Optional) Lưu trạng thái mô hình sau mỗi epoch, dù có cải thiện hay không
        torch.save(model.state_dict(), 'last_model.pth')
        print("Saved last_model.pth for current epoch.")

except KeyboardInterrupt:
    print("Huấn luyện bị gián đoạn. Lưu mô hình hiện tại...")
    torch.save(model.state_dict(), 'model_checkpoint.pth')
    print("Saved model_checkpoint.pth.")
    exit(0)

print("\nHuấn luyện hoàn thành!")

# Đánh giá mô hình trên tập test
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_correct = 0
test_total = 0

with torch.no_grad():
    for images, landmarks, labels in test_loader:
        images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
        outputs = model(images, landmarks)
        _, predicted = torch.max(outputs.data, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")

from sklearn.metrics import confusion_matrix, classification_report

# Tạo danh sách để lưu nhãn thật và nhãn dự đoán
true_labels = []
pred_labels = []

model.eval()
with torch.no_grad():
    for images, landmarks, labels in test_loader:
        images, landmarks, labels = images.to(device), landmarks.to(device), labels.to(device)
        outputs = model(images, landmarks)
        _, predicted = torch.max(outputs.data, 1)
        true_labels.extend(labels.cpu().numpy())
        pred_labels.extend(predicted.cpu().numpy())

# Tính confusion matrix
cm = confusion_matrix(true_labels, pred_labels)
print("Confusion Matrix:")
print(cm)

# Hiển thị F1-score, Precision, Recall cho từng lớp
target_names = [dataset.idx_to_class[i] for i in range(num_classes)]
report = classification_report(true_labels, pred_labels, target_names=target_names)
print("Classification Report:")
print(report)