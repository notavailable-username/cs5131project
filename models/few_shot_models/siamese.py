import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class SiameseEncoder(nn.Module):
    """Siamese network encoder that generates embeddings for similarity comparison"""
    def __init__(self):
        super(SiameseEncoder, self).__init__()
        # Feature extraction backbone
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Embedding layer to produce fixed-size embeddings
        self.fc = nn.Linear(128 * 28 * 28, 256)
    
    def forward(self, x):
        """Extract features and produce embeddings"""
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        x = self.pool(F.relu(self.conv3(x)))  # 56 -> 28
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)  # Create embedding
        return embedding

class SiameseModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        # Initialize Siamese network encoder
        self.encoder = SiameseEncoder()
        self.encoder.to(self.device)
        self.encoder.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Store support example embeddings by class
        self.support_embeddings = {}
        
    def train_from_examples(self, support_examples):
        """
        Compute and store embeddings for each support example
        
        Args:
            support_examples (dict): Dictionary of class_name -> list of examples
        """
        self.support_embeddings = {}
        
        for class_name, examples in support_examples.items():
            if len(examples) == 0:
                continue
                
            # Compute embeddings for this class
            class_embeddings = []
            
            for img in examples:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                
                # Extract embedding
                with torch.no_grad():
                    embedding = self.encoder(img_tensor)
                    class_embeddings.append(embedding)
            
            # Store embeddings for this class
            if class_embeddings:
                self.support_embeddings[class_name] = torch.cat(class_embeddings, dim=0)
        
        print(f"SiameseModel prepared with {len(self.support_embeddings)} classes")
        return True
    
    def predict(self, image_patch):
        """
        Use Siamese network to find the most similar support example
        and return its class with a confidence score
        """
        if not self.support_embeddings:
            return "unknown", 0.0
            
        # Process query image
        image_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = self.encoder(input_tensor)
        
        # Compare with class embeddings using cosine similarity
        max_similarity = -float('inf')
        best_class = "unknown"
        
        for class_name, embeddings in self.support_embeddings.items():
            # Calculate pairwise similarities with all examples of this class
            similarities = F.cosine_similarity(
                query_embedding.unsqueeze(1), 
                embeddings.unsqueeze(0), 
                dim=2
            )
            
            # Get max similarity for this class
            class_similarity = torch.max(similarities).item()
            
            if class_similarity > max_similarity:
                max_similarity = class_similarity
                best_class = class_name
        
        # Convert similarity to confidence score [0,1]
        confidence = (max_similarity + 1) / 2
        
        return best_class, confidence

if __name__ == "__main__":
    import cv2
    import os
    import sys
    
    print("Testing SiameseModel")
    
    # Use a test image if provided as an argument, otherwise just initialize the model
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        test_image_path = sys.argv[1]
        print(f"Using test image: {test_image_path}")
        
        test_image = cv2.imread(test_image_path)
        if test_image is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = SiameseModel(device=device)
            pred, conf = model.predict(test_image)
            print("SiameseModel Prediction:", pred, "Confidence:", conf)
        else:
            print(f"Failed to load test image: {test_image_path}")
    else:
        print("No test image provided. Initializing model only.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = SiameseModel(device=device)
        print("Model initialized successfully")
