import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class MiniImageNet(Dataset):
    def __init__(self, root, split='train', transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        # Load CSV file
        csv_path = os.path.join(root, f"{split}.csv")
        self.data = pd.read_csv(csv_path)
        print(self.data)

        # Extract image names and labels
        self.image_paths = self.data['filename'].tolist()
        self.labels = self.data['label'].tolist()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, self.image_paths[idx])
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# Define image transformations (normalize & resize for CNNs)
transform = transforms.Compose([
    transforms.Resize((84, 84)),  # Resize to Mini-ImageNet size
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
