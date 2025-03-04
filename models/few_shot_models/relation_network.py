import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2

class RelationNetworkCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(RelationNetworkCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Two poolings: 224 -> 112 -> 56
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class RelationNetworkModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.num_classes = 5
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        self.model = RelationNetworkCNN(num_classes=self.num_classes)
        self.model.to(self.device)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
    
    def predict(self, image_patch):
        image_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
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
        model = RelationNetworkModel(device=device)
        pred, conf = model.predict(dummy_patch)
        print("RelationNetworkModel Prediction:", pred, "Confidence:", conf)
    else:
        print("Dummy image not found.")
