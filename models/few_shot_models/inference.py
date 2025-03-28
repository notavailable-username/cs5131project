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
        # Process input images
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
        
        # Extract features or get predictions
        n_way = len(class_names)
        n_shots = [len(support_paths[class_name]) for class_name in class_names]
        
        if finetune:
            # Fine-tuning on support examples
            # Create a copy of the model for fine-tuning
            model_copy = self._create_model_copy()
            self._finetune_model(model_copy, support_images, support_labels, n_way, finetune_steps, finetune_lr)
            
            # Get predictions using fine-tuned model
            results = self._predict_with_model(model_copy, query_images, query_paths, class_names)
        else:
            # Get predictions without fine-tuning
            if self.is_meta_baseline and hasattr(self.model, 'forward'):
                # Use meta-baseline model directly if possible
                results = self._predict_with_meta_baseline(support_images, support_labels, query_images, query_paths, class_names)
            else:
                # Use the standard prototype-based approach
                results = self._predict_with_prototypes(support_images, support_labels, query_images, query_paths, class_names)
            
        return results
    
    def _create_model_copy(self):
        """Create a copy of the model for fine-tuning."""
        if self.is_meta_baseline:
            # For meta-baseline model
            model_copy = type(self.model)(self.model.encoder, self.model.method, self.model.temp.item())
            model_copy.load_state_dict(self.model.state_dict())
        else:
            # For classifier model
            model_copy = type(self.model)()
            model_copy.load_state_dict(self.model.state_dict())
        
        model_copy = model_copy.to(self.device)
        return model_copy
    
    def _finetune_model(self, model, support_images, support_labels, n_way, steps, lr):
        """Fine-tune model on support examples."""
        # Create one-hot encoded labels
        support_labels_onehot = F.one_hot(support_labels, n_way).float()
        
        # Set model to training mode
        model.train()
        
        # Setup optimizer - only fine-tune the last layer for simplicity
        if hasattr(model, 'encoder'):
            # For meta-baseline model
            params = [p for p in model.parameters() if p.requires_grad]
        else:
            # For standard model
            params = model.parameters()
        
        optimizer = torch.optim.Adam(params, lr=lr)
        
        # Fine-tuning loop
        for step in range(steps):
            optimizer.zero_grad()
            
            # Forward pass
            if self.is_meta_baseline:
                # For meta-baseline model - extract features first
                with torch.no_grad():
                    features = model.encoder(support_images)
                
                # Compute prototypes
                prototypes = utils.few_shot.compute_prototypes(features, support_labels_onehot)
                
                # Compute logits
                logits = utils.few_shot.compute_logits(features, prototypes)
            else:
                # For standard model - get features directly
                features = model(support_images)
                
                # Compute prototypes
                prototypes = utils.few_shot.compute_prototypes(features, support_labels_onehot)
                
                # Compute logits
                logits = utils.few_shot.compute_logits(features, prototypes)
            
            # Compute loss
            loss = F.cross_entropy(logits, support_labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if step % 5 == 0:
                print(f"Fine-tuning step {step}, loss: {loss.item():.4f}")
        
        # Set model back to evaluation mode
        model.eval()
    
    def _predict_with_meta_baseline(self, support_images, support_labels, query_images, query_paths, class_names):
        """Predict using meta-baseline model."""
        n_way = len(class_names)
        n_support = support_images.shape[0]
        n_query = query_images.shape[0]
        
        # Prepare inputs in the expected format for meta-baseline
        # Reshape support images to [n_way, n_shot, C, H, W]
        support_by_class = []
        for i in range(n_way):
            support_by_class.append(support_images[support_labels == i])
            
        # Get the minimum number of examples per class
        min_n_shot = min(len(class_examples) for class_examples in support_by_class)
        
        # Trim to same number of shots per class
        x_shot = torch.stack([examples[:min_n_shot] for examples in support_by_class])
        
        # Create query tensor
        x_query = query_images
        
        with torch.no_grad():
            # Use the meta-baseline model's forward function
            logits = self.model(x_shot, x_query)
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
    
    def _predict_with_prototypes(self, support_images, support_labels, query_images, query_paths, class_names):
        """Predict using prototype-based approach."""
        n_way = len(class_names)
        
        with torch.no_grad():
            # Extract features
            if hasattr(self.model, 'encoder'):
                support_features = self.model.encoder(support_images)
                query_features = self.model.encoder(query_images)
            else:
                support_features = self.model(support_images)
                query_features = self.model(query_images)
            
            # Compute prototypes for each class
            prototypes = torch.zeros(n_way, support_features.shape[1], device=self.device)
            for i in range(n_way):
                mask = support_labels == i
                if not mask.any():
                    raise ValueError(f"No support examples for class {i}")
                prototypes[i] = support_features[mask].mean(dim=0)
            
            # Compute logits as negative distances
            logits = -utils.few_shot.compute_distance(query_features, prototypes)
            
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

    def _predict_with_model(self, model, query_images, query_paths, class_names):
        """Predict using a (potentially fine-tuned) model."""
        with torch.no_grad():
            if hasattr(model, 'encoder'):
                features = model.encoder(query_images)
            else:
                features = model(query_images)
            
            # Using the model's classification layer if it exists, otherwise logits will be the features
            if hasattr(model, 'forward'):
                logits = model.forward_fc(features)
            else:
                # We'll compute similarities to centroids of each class from training
                logits = features
            
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
