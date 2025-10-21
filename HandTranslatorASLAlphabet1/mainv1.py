from PIL import Image
import torch
from torchvision import transforms
from model import ResNet

# Load model
num_classes = 29
model = ResNet(num_classes)
model.load_state_dict(torch.load('best_model.pth', map_location='cpu'))
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# Dự đoán ảnh tay đã crop
img_path = r"D:\DoAn\Code1\HandTranslatorASLAlphabet\k2.jpg"
img = Image.open(img_path).convert("RGB")
input_tensor = preprocess(img).unsqueeze(0)

with torch.no_grad():
    output = model(input_tensor)
    pred = torch.argmax(output, dim=1).item()

gestos = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F',
          6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L',
          12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R',
          18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X',
          24: 'Y', 25: 'Z', 26: 'del', 27: 'nothing', 28: 'space'}

print(" Ký tự nhận diện được:", gestos[pred])
