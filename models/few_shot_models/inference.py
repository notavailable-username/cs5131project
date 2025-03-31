import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
import models
import utils
import utils.few_shot as fs
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union


class FewShotPredictor:
    """
    A class for making few-shot learning predictions based on support and query images.
    This uses the Meta-Baseline architecture for few-shot classification.
    """
    
    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize the few-shot predictor.
        
        Args:
            checkpoint_path: Path to the model checkpoint (.pth file)
            device: Device to run the model on ('cpu' or 'cuda'). If None, use CUDA if available.
        """
        self.checkpoint_path = checkpoint_path
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        
        # Load model
        self._load_model()
        
        # Define transforms for image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                std=[0.229, 0.224, 0.225])
        ])
    
    def _load_model(self):
        """Load the model from checkpoint."""
        print(f"Loading model from {self.checkpoint_path}")
        model_dict = torch.load(self.checkpoint_path, map_location=self.device)
        
        # Check if it's a meta-baseline model or a classifier model
        if model_dict.get('model') == 'meta-baseline':
            # Meta-baseline model
            self.model = models.load(model_dict)
            self.is_meta_baseline = True
        else:
            # Classifier model - we'll use its encoder
            encoder_name = model_dict['model_args']['encoder']
            encoder_args = model_dict['model_args'].get('encoder_args', {})
            
            # Create encoder
            self.model = models.make(encoder_name, **encoder_args)
            self.model.load_state_dict(model_dict['model_sd'], strict=False)
            self.is_meta_baseline = False
        
        self.model.to(self.device)
        self.model.eval()
        print("Model loaded successfully")
    
    def _load_image(self, image_path: str) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Preprocessed image tensor
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        image = Image.open(image_path).convert('RGB')
        return self.transform(image)
    
    def predict(self, 
                support_paths: Dict[str, List[str]], 
                query_paths: List[str],
                finetune: bool = False,
                finetune_steps: int = 10,
                finetune_lr: float = 0.01) -> List[Dict[str, Union[str, float]]]:
        """
        Predict classes for query images based on support images.
        
        Args:
            support_paths: Dictionary mapping class names to lists of support image paths
            query_paths: List of paths to query images
            finetune: Whether to finetune the model on support examples
            finetune_steps: Number of fine-tuning steps (if finetune=True)
            finetune_lr: Learning rate for fine-tuning (if finetune=True)
            
        Returns:
            List of dictionaries containing predicted class and confidence for each query image
        """
        # Process support images and labels
        support_images = []
        support_labels = []
        class_names = list(support_paths.keys())
        
        for class_idx, class_name in enumerate(class_names):
            for path in support_paths[class_name]:
                img = self._load_image(path)
                support_images.append(img)
                support_labels.append(class_idx)
                
        if not support_images:
            raise ValueError("No support images provided")
        
        # Stack images and convert labels to tensor
        support_images = torch.stack(support_images).to(self.device)
        support_labels = torch.tensor(support_labels, device=self.device)
        
        # Process query images
        query_images = []
        for path in query_paths:
            img = self._load_image(path)
            query_images.append(img)
        
        query_images = torch.stack(query_images).to(self.device)
        
        # One-hot encode support labels
        n_way = len(class_names)
        support_labels_onehot = F.one_hot(support_labels, n_way).float()
        
        # Make predictions using a single unified approach
        with torch.no_grad():
            if finetune:
                # Fine-tuning mode: create a working copy of the model
                if self.is_meta_baseline:
                    model = type(self.model)(self.model.encoder, self.model.method, self.model.temp.item())
                    model.load_state_dict(self.model.state_dict())
                else:
                    model = type(self.model)()
                    model.load_state_dict(self.model.state_dict())
                
                model = model.to(self.device)
                model.train()
                
                # Create optimizer
                if hasattr(model, 'encoder'):
                    params = [p for p in model.parameters() if p.requires_grad]
                else:
                    params = model.parameters()
                
                optimizer = torch.optim.Adam(params, lr=finetune_lr)
                
                # Fine-tuning loop
                for step in range(finetune_steps):
                    optimizer.zero_grad()
                    
                    # Extract features
                    if hasattr(model, 'encoder'):
                        features = model.encoder(support_images)
                    else:
                        features = model(support_images)
                    
                    # Compute prototypes
                    prototypes = fs.compute_prototypes(features, support_labels_onehot)
                    
                    # Compute logits
                    temp = model.temp if hasattr(model, 'temp') else 1.0
                    logits = utils.compute_logits(features, prototypes, 'cos', temp)
                    
                    # Compute loss
                    loss = F.cross_entropy(logits, support_labels)
                    
                    # Release graph for manual backward
                    with torch.set_grad_enabled(True):
                        loss.backward()
                        optimizer.step()
                    
                    if step % 5 == 0:
                        print(f"Fine-tuning step {step}, loss: {loss.item():.4f}")
                
                # Set model back to evaluation mode
                model.eval()
            else:
                # Non-fine-tuning mode: use the original model
                model = self.model
            
            # Evaluation/prediction phase
            
            # Special handling for meta-baseline model
            if self.is_meta_baseline and hasattr(model, 'forward'):
                # Organize support images by class for meta-baseline's episodic format
                support_by_class = []
                for i in range(n_way):
                    class_images = support_images[support_labels == i]
                    if len(class_images) > 0:
                        support_by_class.append(class_images)
                
                # Use balanced shots for meta-baseline by taking min shot count per class
                if len(support_by_class) == n_way:
                    min_n_shot = min(len(class_examples) for class_examples in support_by_class)
                    x_shot = torch.stack([examples[:min_n_shot] for examples in support_by_class])
                    
                    # Meta-baseline forward pass with support and query
                    logits = model(x_shot, query_images)
                else:
                    # Fallback to prototype-based approach if classes are missing
                    print(f"Warning: Some classes have no support examples. Using prototype approach instead.")
                    
                    if hasattr(model, 'encoder'):
                        support_features = model.encoder(support_images)
                        query_features = model.encoder(query_images)
                    else:
                        support_features = model(support_images)
                        query_features = model(query_images)
                    
                    prototypes = fs.compute_prototypes(support_features, support_labels_onehot)
                    temp = model.temp if hasattr(model, 'temp') else 1.0
                    logits = utils.compute_logits(query_features, prototypes, 'cos', temp)
            else:
                # Standard feature extraction and prototype-based classification
                if hasattr(model, 'encoder'):
                    support_features = model.encoder(support_images)
                    query_features = model.encoder(query_images)
                else:
                    support_features = model(support_images)
                    query_features = model(query_images)
                
                prototypes = fs.compute_prototypes(support_features, support_labels_onehot)
                temp = model.temp if hasattr(model, 'temp') else 1.0
                logits = utils.compute_logits(query_features, prototypes, 'cos', temp)
                
            # Apply softmax to get confidence values
            probabilities = F.softmax(logits, dim=1)
        
        # Process results
        results = []
        for i, query_path in enumerate(query_paths):
            pred_idx = logits[i].argmax().item()
            pred_class = class_names[pred_idx]
            conf_value = probabilities[i, pred_idx].item()
            
            results.append({
                'image_path': query_path,
                'predicted_class': pred_class,
                'confidence': conf_value,
                'all_confidences': {class_name: probabilities[i, j].item() 
                                    for j, class_name in enumerate(class_names)}
            })
        
        return results


# Example usage
if __name__ == "__main__":
    # Example usage of the FewShotPredictor
    predictor = FewShotPredictor(
        checkpoint_path="./save/meta_mini-imagenet-1shot_meta-baseline-resnet12/max-va.pth")
    
    # Define support images for each class
    support_paths = {
        "dog": [
            "./materials/mini-imagenet/images/n02110063_2414.jpg", 
            "./materials/mini-imagenet/images/n02110063_1234.jpg"
        ],
        "cat": [
            "./materials/mini-imagenet/images/n02123159_1950.jpg", 
            "./materials/mini-imagenet/images/n02123159_5901.jpg"
        ]
    }
    
    # Define query images
    query_paths = [
        "./materials/mini-imagenet/images/n02110063_6974.jpg",  # a dog
        "./materials/mini-imagenet/images/n02123159_9803.jpg"   # a cat
    ]
    
    # Get predictions without fine-tuning
    results = predictor.predict(support_paths, query_paths)
    print("Results without fine-tuning:")
    for result in results:
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print()
    
    # Get predictions with fine-tuning
    results_finetuned = predictor.predict(support_paths, query_paths, finetune=True, finetune_steps=20, finetune_lr=0.01)
    print("\nResults with fine-tuning:")
    for result in results_finetuned:
        print(f"Image: {os.path.basename(result['image_path'])}")
        print(f"Predicted class: {result['predicted_class']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print()
