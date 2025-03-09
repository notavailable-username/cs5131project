import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2
import copy

class MAMLEncoder(nn.Module):
    """Encoder network for MAML with parameters that can quickly adapt"""
    def __init__(self):
        super(MAMLEncoder, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) 
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2, 2)
        # Two pooling layers: 224 -> 112 -> 56
        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 5)  # output dimensionality can be adjusted
    
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  # 224 -> 112
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  # 112 -> 56
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

class MAMLModel:
    def __init__(self, device='cpu', inner_lr=0.01, adapt_steps=5):
        self.device = device
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        
        # Meta-model that can quickly adapt
        self.model = MAMLEncoder()
        self.model.to(self.device)
        self.model.eval()
        
        # Inner loop adaptation parameters
        self.inner_lr = inner_lr
        self.adapt_steps = adapt_steps
        
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406],
                                 std=[0.229,0.224,0.225])
        ])
        
        # Store class mapping for adaptation
        self.class_to_idx = {}
        self.adapted_models = {}
    
    def train_from_examples(self, support_examples):
        """
        Create adapted models for each class by fine-tuning with support examples
        
        Args:
            support_examples (dict): Dictionary of class_name -> list of examples
        """
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.adapted_models = {}
        
        for class_name, examples in support_examples.items():
            if len(examples) < 2:  # Need at least a few examples for adaptation
                continue
            
            # Create a specialized model for this class
            adapted_model = copy.deepcopy(self.model)
            adapted_model.train()
            
            # Extract support set tensors
            support_tensors = []
            for img in examples:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = self.transform(img_rgb).unsqueeze(0).to(self.device)
                support_tensors.append(img_tensor)
            
            # Combine tensors and create targets (all same class)
            support_images = torch.cat(support_tensors, dim=0)
            class_idx = self.class_to_idx.get(class_name, len(self.classes)-1)  # Default to unknown
            support_labels = torch.tensor([class_idx] * len(support_tensors)).to(self.device)
            
            # Perform inner loop adaptation
            for step in range(self.adapt_steps):
                # Forward pass
                logits = adapted_model(support_images)
                loss = F.cross_entropy(logits, support_labels)
                
                # Compute gradients and update model
                grads = torch.autograd.grad(loss, adapted_model.parameters())
                
                # Manual SGD update
                for p, g in zip(adapted_model.parameters(), grads):
                    p.data.sub_(self.inner_lr * g)
            
            # Store the adapted model
            adapted_model.eval()
            self.adapted_models[class_name] = adapted_model
        
        print(f"MAMLModel created {len(self.adapted_models)} adapted models")
        return True
    
    def predict(self, image_patch):
        """
        Use adapted models to make predictions
        """
        if not self.adapted_models:
            return "unknown", 0.0
        
        # Process query image
        image_rgb = cv2.cvtColor(image_patch, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(image_rgb).unsqueeze(0).to(self.device)
        
        best_confidence = -float('inf')
        best_class = "unknown"
        
        # Try each adapted model
        for class_name, adapted_model in self.adapted_models.items():
            with torch.no_grad():
                logits = adapted_model(input_tensor)
                probabilities = F.softmax(logits, dim=1)
                
                # Get probability for the class this model was adapted for
                class_idx = self.class_to_idx.get(class_name, len(self.classes)-1)
                confidence = probabilities[0, class_idx].item()
                
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_class = class_name
        
        return best_class, best_confidence

if __name__ == "__main__":
    import cv2
    import os
    import sys
    
    print("Testing MAMLModel")
    
    # Use a test image if provided as an argument, otherwise just initialize the model
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        test_image_path = sys.argv[1]
        print(f"Using test image: {test_image_path}")
        
        test_image = cv2.imread(test_image_path)
        if test_image is not None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = MAMLModel(device=device)
            pred, conf = model.predict(test_image)
            print("MAMLModel Prediction:", pred, "Confidence:", conf)
        else:
            print(f"Failed to load test image: {test_image_path}")
    else:
        print("No test image provided. Initializing model only.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = MAMLModel(device=device)
        print("Model initialized successfully")
