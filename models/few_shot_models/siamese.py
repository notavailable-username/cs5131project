import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

class SiameseCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(SiameseCNN, self).__init__()
        # Simple CNN backbone (for demonstration)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Assuming input image size 224x224, after 3 poolings: 28x28 feature map
        self.fc1 = nn.Linear(128 * 28 * 28, 256)
        self.fc2 = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class SiameseModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.num_classes = 5
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        self.model = SiameseCNN(num_classes=self.num_classes)
        # In a full solution, load pretrained weights here.
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    
    def predict(self, image_patch):
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb)
        input_tensor = input_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            confidence, pred_idx = torch.max(probs, dim=1)
        predicted_class = self.classes[pred_idx.item()]
        return predicted_class, confidence.item()

if __name__ == "__main__":
    import cv2
    dummy_patch = cv2.imread("dummy.jpg")
    if dummy_patch is not None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SiameseModel(device=device)
        pred, conf = model.predict(dummy_patch)
        print("SiameseModel Prediction:", pred, "Confidence:", conf)
    else:
        print("Dummy image not found.")
