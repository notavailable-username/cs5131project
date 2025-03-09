import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np

class ProtoEncoder(nn.Module):
    """Encoder network for Prototypical Networks"""
    def __init__(self):
        super(ProtoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2, 2)
        # 4 pooling layers: 224 -> 112 -> 56 -> 28 -> 14
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        # Create embedding vector
        embedding = x.view(x.size(0), -1)
        return embedding

class PrototypicalModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        self.encoder = ProtoEncoder()
        self.encoder.to(self.device)
        self.encoder.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Store class prototypes
        self.prototypes = {}
    
    def train_from_examples(self, support_examples):
        """
        Compute class prototypes from support examples
        
        Args:
            support_examples (dict): Dictionary of class_name -> list of examples
        """
        self.prototypes = {}
        
        for class_name, examples in support_examples.items():
            if len(examples) == 0:
                continue
                
            # Extract features for all examples of this class
            class_embeddings = []
            
            for img in examples:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.encoder(img_tensor)
                    class_embeddings.append(embedding)
            
            # Compute prototype as mean of all examples
            if class_embeddings:
                class_prototype = torch.mean(torch.cat(class_embeddings), dim=0, keepdim=True)
                self.prototypes[class_name] = class_prototype
        
        print(f"PrototypicalModel prepared with {len(self.prototypes)} class prototypes")
        return True
    
    def predict(self, image_patch):
        """
        Predict class by finding the nearest prototype using Euclidean distance
        """
        if not self.prototypes:
            return "unknown", 0.0
        
        # Process query image
        image_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = self.encoder(input_tensor)
        
        # Calculate distances to all prototypes
        min_distance = float('inf')
        best_class = "unknown"
        distances = {}
        
        for class_name, prototype in self.prototypes.items():
            # Euclidean distance (squared)
            distance = torch.sum((query_embedding - prototype)**2).item()
            distances[class_name] = distance
            
            if distance < min_distance:
                min_distance = distance
                best_class = class_name
        
        # Convert distance to confidence (closer = higher confidence)
        if len(distances) > 1:
            # Calculate confidence as softmax over negative distances
            neg_distances = [-d for d in distances.values()]
            exp_neg_dist = [np.exp(d) for d in neg_distances]
            sum_exp = sum(exp_neg_dist)
            confidence = np.exp(-min_distance) / sum_exp
        else:
            # If only one class, use a simple distance-based confidence
            confidence = np.exp(-min_distance / 10.0)  # Scale for reasonable values
        
        return best_class, confidence

if __name__ == "__main__":
    import cv2
    import os
    import sys
    
    print("Testing PrototypicalModel")
    
    # Use a test image if provided as an argument, otherwise just initialize the model
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        test_image_path = sys.argv[1]
        print(f"Using test image: {test_image_path}")
        
        test_image = cv2.imread(test_image_path)
        if test_image is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = PrototypicalModel(device=device)
            pred, conf = model.predict(test_image)
            print("PrototypicalModel Prediction:", pred, "Confidence:", conf)
        else:
            print(f"Failed to load test image: {test_image_path}")
    else:
        print("No test image provided. Initializing model only.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = PrototypicalModel(device=device)
        print("Model initialized successfully")
