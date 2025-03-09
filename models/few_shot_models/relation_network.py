import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np

class EmbeddingNetwork(nn.Module):
    """Embedding network to extract features from images"""
    def __init__(self):
        super(EmbeddingNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # Two poolings: 224 -> 112 -> 56
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # 224 -> 112
        x = self.pool(F.relu(self.conv2(x)))  # 112 -> 56
        return x  # Output feature maps, not flattened

class RelationNetwork(nn.Module):
    """Relation Network to compute similarity between query and support examples"""
    def __init__(self):
        super(RelationNetwork, self).__init__()
        # Processes concatenated feature maps
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 1)
        self.pool = nn.MaxPool2d(2, 2)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))  # Score between 0 and 1
        return x

class RelationNetworkModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        
        # Initialize embedding and relation networks
        self.embedding_net = EmbeddingNetwork()
        self.relation_net = RelationNetwork()
        
        self.embedding_net.to(self.device)
        self.relation_net.to(self.device)
        
        self.embedding_net.eval()
        self.relation_net.eval()
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        
        # Storage for support set embeddings
        self.support_features = {}
    
    def train_from_examples(self, support_examples):
        """
        Process support examples and store their embeddings
        
        Args:
            support_examples (dict): Dictionary of class_name -> list of examples
        """
        self.support_features = {}
        
        for class_name, examples in support_examples.items():
            if len(examples) == 0:
                continue
                
            class_embeddings = []
            for img in examples:
                # Convert and preprocess image
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                
                # Extract features using embedding network
                with torch.no_grad():
                    embedding = self.embedding_net(img_tensor)
                    class_embeddings.append(embedding)
            
            if class_embeddings:
                # Store all embeddings for this class
                self.support_features[class_name] = class_embeddings
        
        print(f"RelationNetworkModel prepared with {len(self.support_features)} classes")
        return True
    
    def predict(self, image_patch):
        """
        Use relation network to compare query with support examples
        """
        if not self.support_features:
            return "unknown", 0.0
        
        # Process query image
        image_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        query_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Extract query features
        with torch.no_grad():
            query_features = self.embedding_net(query_tensor)
        
        best_score = -float('inf')
        best_class = "unknown"
        class_scores = {}
        
        # Compare with each class using relation network
        for class_name, support_embeddings in self.support_features.items():
            class_score = 0.0
            count = 0
            
            # Compare with each support example of this class
            for support_embedding in support_embeddings:
                # Concatenate features along channel dimension
                paired_features = torch.cat([query_features, support_embedding], dim=1)
                
                # Calculate relation score
                relation_score = self.relation_net(paired_features)
                class_score += relation_score.item()
                count += 1
            
            # Average score for this class
            if count > 0:
                avg_score = class_score / count
                class_scores[class_name] = avg_score
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_class = class_name
        
        confidence = best_score if best_score > 0 else 0.0
        
        return best_class, confidence

if __name__ == "__main__":
    import cv2
    import os
    import sys
    
    print("Testing RelationNetworkModel")
    
    # Use a test image if provided as an argument, otherwise just initialize the model
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        test_image_path = sys.argv[1]
        print(f"Using test image: {test_image_path}")
        
        test_image = cv2.imread(test_image_path)
        if test_image is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = RelationNetworkModel(device=device)
            pred, conf = model.predict(test_image)
            print("RelationNetworkModel Prediction:", pred, "Confidence:", conf)
        else:
            print(f"Failed to load test image: {test_image_path}")
    else:
        print("No test image provided. Initializing model only.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = RelationNetworkModel(device=device)
        print("Model initialized successfully")
