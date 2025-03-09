import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import numpy as np

class MatchingEncoder(nn.Module):
    """Encoder network for Matching Networks"""
    def __init__(self):
        super(MatchingEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Fully connected embedding
        self.fc = nn.Linear(128 * 28 * 28, 256)
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112 -> 56
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # 56 -> 28
        x = x.view(x.size(0), -1)
        embedding = self.fc(x)
        return embedding

class AttentionModel(nn.Module):
    """Attention model for weighting support set examples"""
    def __init__(self):
        super(AttentionModel, self).__init__()
        
    def forward(self, query_embedding, support_embeddings):
        """
        Computes attention weights based on cosine similarity
        
        Args:
            query_embedding: Tensor of shape [1, embedding_dim]
            support_embeddings: Tensor of shape [n_support, embedding_dim]
            
        Returns:
            attention_weights: Tensor of shape [n_support]
        """
        # Compute cosine similarity between query and each support example
        similarities = F.cosine_similarity(
            query_embedding.unsqueeze(1), 
            support_embeddings.unsqueeze(0),
            dim=2
        ).squeeze(0)
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(similarities, dim=0)
        return attention_weights

class MatchingModel:
    def __init__(self, device='cpu'):
        self.device = device
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        
        # Initialize encoder network
        self.encoder = MatchingEncoder()
        self.encoder.to(self.device)
        self.encoder.eval()
        
        # Initialize attention model
        self.attention = AttentionModel()
        self.attention.to(self.device)
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        # Support set storage
        self.support_embeddings = {}
        self.support_labels = {}
    
    def train_from_examples(self, support_examples):
        """
        Process support examples and store their embeddings
        
        Args:
            support_examples (dict): Dictionary of class_name -> list of examples
        """
        all_embeddings = []
        all_labels = []
        
        for class_name, examples in support_examples.items():
            if len(examples) == 0:
                continue
                
            for img in examples:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.encoder(img_tensor)
                    all_embeddings.append(embedding)
                    all_labels.append(class_name)
        
        if all_embeddings:
            self.support_embeddings = torch.cat(all_embeddings)
            self.support_labels = all_labels
            
        print(f"MatchingModel prepared with {len(all_labels)} support examples")
        return True
    
    def predict(self, image_patch):
        """
        Use attention-weighted nearest neighbors for classification
        """
        if not hasattr(self, 'support_embeddings') or len(self.support_labels) == 0:
            return "unknown", 0.0
        
        # Process query image
        image_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        # Get query embedding
        with torch.no_grad():
            query_embedding = self.encoder(input_tensor)
            
            # Compute attention weights
            attention_weights = self.attention(query_embedding, self.support_embeddings)
            
            # Weighted class prediction
            class_probs = {}
            for i, label in enumerate(self.support_labels):
                if label not in class_probs:
                    class_probs[label] = 0
                class_probs[label] += attention_weights[i].item()
            
            # Get most probable class
            best_class = max(class_probs.keys(), key=lambda k: class_probs[k])
            confidence = class_probs[best_class]
        
        return best_class, confidence

if __name__ == "__main__":
    import cv2
    import os
    import sys
    
    print("Testing MatchingModel")
    
    # Use a test image if provided as an argument, otherwise just initialize the model
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        test_image_path = sys.argv[1]
        print(f"Using test image: {test_image_path}")
        
        test_image = cv2.imread(test_image_path)
        if test_image is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MatchingModel(device=device)
            pred, conf = model.predict(test_image)
            print("MatchingModel Prediction:", pred, "Confidence:", conf)
        else:
            print(f"Failed to load test image: {test_image_path}")
    else:
        print("No test image provided. Initializing model only.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MatchingModel(device=device)
        print("Model initialized successfully")
