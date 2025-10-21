import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights  # Sử dụng API mới
from torch.utils.data import Dataset
from PIL import Image
import os

# Definición de la arquitectura ResNet
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # Tải ResNet18 với trọng số được huấn luyện sẵn
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        num_ftrs = self.resnet.fc.in_features
        
        # Xóa lớp fully connected cuối cùng
        self.resnet.fc = nn.Identity()
        
        # Thêm các lớp xử lý landmark
        self.landmark_fc = nn.Sequential(
            nn.Linear(63, 128),  # 21 điểm landmark * 3 tọa độ (x,y,z)
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        
        # Lớp phân loại cuối cùng
        self.fc = nn.Linear(num_ftrs + 128, num_classes)

    def forward(self, x):
        # Khi dự đoán, chỉ sử dụng đặc trưng ảnh
        x = self.resnet(x)
        # Tạo tensor 0 cho phần landmark features
        batch_size = x.shape[0]
        landmark_features = torch.zeros(batch_size, 128).to(x.device)
        # Ghép đặc trưng ảnh và landmark
        combined = torch.cat((x, landmark_features), dim=1)
        return self.fc(combined)

# Dataset personalizado para cargar las imágenes
class ASLDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_list = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Obtener la lista de categorías (subdirectorios)
        categories = os.listdir(data_dir)
        categories.sort()  # Ordenar alfabéticamente para asegurar consistencia

        for i, category in enumerate(categories):
            category_path = os.path.join(data_dir, category)
            if os.path.isdir(category_path):
                for file_name in os.listdir(category_path):
                    file_path = os.path.join(category_path, file_name)
                    self.file_list.append((file_path, i))
                self.class_to_idx[category] = i
                self.idx_to_class[i] = category

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, label = self.file_list[idx]
        # Garantizar que la imagen esté en formato RGB
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
