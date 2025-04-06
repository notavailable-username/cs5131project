import sys
import os
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from collections import defaultdict
from typing import List, Dict, Tuple, Optional, Union
import datetime

from models.few_shot_models.models.models import make, load

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
            self.model = load(model_dict)
            self.is_meta_baseline = True
        else:
            # Classifier model - we'll use its encoder
            encoder_name = model_dict['model_args']['encoder']
            encoder_args = model_dict['model_args'].get('encoder_args', {})
            
            # Create encoder
            self.model = make(encoder_name, **encoder_args)
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
        print(f"[PREDICT] Starting prediction with {len(support_paths)} classes and {len(query_paths)} query images")
        
        # Process query images
        print(f"[PREDICT] Processing {len(query_paths)} query images...")
        query_images = []
        for path in query_paths:
            img = self._load_image(path)
            query_images.append(img)
        
        query_images = torch.stack(query_images).to(self.device)
        print(f"[PREDICT] Query images processed and moved to {self.device}")
        
        # Process support images and organize by class
        print(f"[PREDICT] Processing support images...")
        class_names = list(support_paths.keys())
        n_way = len(class_names)
        
        # Load all support images by class
        class_tensors = []
        for class_idx, class_name in enumerate(class_names):
            print(f"[PREDICT]   - Loading {len(support_paths[class_name])} images for class '{class_name}'")
            class_images = []
            for path in support_paths[class_name]:
                img = self._load_image(path)
                class_images.append(img)
            
            if not class_images:
                raise ValueError(f"No support images provided for class '{class_name}'")
                
            # Stack all images for this class
            class_tensor = torch.stack(class_images).to(self.device)
            class_tensors.append(class_tensor)
        
        if not class_tensors:
            raise ValueError("No support images provided")
        
        # Instead of duplicating samples in input space, we can:
        # 1. Process each class separately to get features
        # 2. Compute prototypes in feature space
        # 3. Use these prototypes for classification
        # This approach doesn't work with meta-baseline's expected input format
        
        # For meta-baseline, we need consistent shot count. We'll use uniform sampling
        # to avoid biasing the prototypes toward specific samples
        shot_counts = [tensor.size(0) for tensor in class_tensors]
        max_shots = max(shot_counts)
        min_shots = min(shot_counts)
        print(f"[PREDICT] Support images per class: min={min_shots}, max={max_shots}")
        
        if min_shots != max_shots:
            print(f"[PREDICT] Balancing support shots across classes using uniform sampling")
            
            balanced_tensors = []
            for tensor in class_tensors:
                n_shots = tensor.size(0)
                if n_shots < max_shots:
                    # Create indices that uniformly sample from available shots
                    # This ensures every sample is represented as equally as possible
                    indices = torch.tensor([i % n_shots for i in range(max_shots)], device=self.device)
                    balanced = tensor[indices]
                    balanced_tensors.append(balanced)
                else:
                    balanced_tensors.append(tensor)
            
            x_shot = torch.stack(balanced_tensors)
        else:
            # All classes have the same number of shots, no balancing needed
            x_shot = torch.stack(class_tensors)
        
        print(f"[PREDICT] Support images processed into shape {x_shot.shape}")
        
        # Make predictions
        with torch.no_grad():
            if finetune:
                print(f"[PREDICT] Starting fine-tuning for {finetune_steps} steps with learning rate {finetune_lr}...")
                
                # Create a working copy of the model for fine-tuning
                model = self.model
                model.train()
                
                # Create optimizer for the full model
                optimizer = torch.optim.Adam(model.parameters(), lr=finetune_lr)
                
                # Fine-tuning loop following train_meta.py approach
                for step in range(finetune_steps):
                    optimizer.zero_grad()
                    
                    # Select subset of query images for this step
                    n_query_per_class = min(len(query_images) // n_way, min_shots)
                    total_queries = n_query_per_class * n_way
                    
                    # Sample query images
                    if len(query_images) > total_queries:
                        idx = torch.randperm(len(query_images))[:total_queries]
                        train_query = query_images[idx]
                    else:
                        train_query = query_images
                    
                    # Create labels for query images: [0, 0, ..., 1, 1, ..., n_way-1, n_way-1]
                    # Each class has n_query_per_class examples
                    query_labels = torch.arange(n_way, device=self.device).repeat_interleave(n_query_per_class)
                    
                    # Forward pass
                    with torch.set_grad_enabled(True):
                        # Meta-baseline expects: x_shot [n_way, n_shot, C, H, W], x_query [n_query, C, H, W]
                        logits = model(x_shot, train_query)
                        # Reshape logits if needed to be [n_query, n_way]
                        if len(logits.shape) > 2:
                            logits = logits.reshape(-1, n_way)
                        
                        # Compute loss
                        loss = F.cross_entropy(logits, query_labels)
                        loss.backward()
                        optimizer.step()
                    
                    if step % 5 == 0:
                        print(f"[PREDICT]   Fine-tuning step {step}/{finetune_steps}, loss: {loss.item():.4f}")
                
                # Set model back to evaluation mode after fine-tuning
                model.eval()
                print(f"[PREDICT] Fine-tuning completed")
            else:
                print(f"[PREDICT] Using pre-trained model without fine-tuning")
                model = self.model
            
            # Evaluation/prediction phase
            print(f"[PREDICT] Starting prediction phase...")
            
            # Run the model's forward pass with support and query images
            print(f"[PREDICT] Running forward pass with support and query images")
            logits = model(x_shot, query_images)
            if len(logits.shape) > 2:
                logits = logits.reshape(-1, n_way)
            
            # Apply softmax to get confidence values
            print(f"[PREDICT] Computing probabilities using softmax")
            probabilities = F.softmax(logits, dim=1)
        
        # Process results
        print(f"[PREDICT] Processing results for {len(query_paths)} query images")
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
        
        print(f"[PREDICT] Prediction completed successfully")
        return results


def write_results_to_file(results: List[Dict], output_path: str):
    """
    Write prediction results to a text file.
    
    Args:
        results: List of prediction result dictionaries
        output_path: Path to save the output text file
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write(f"Prediction Results - {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total images processed: {len(results)}\n\n")
        
        for result in results:
            f.write(f"Image: {os.path.basename(result['image_path'])}\n")
            f.write(f"Predicted class: {result['predicted_class']}\n")
            f.write(f"Confidence: {result['confidence']:.4f}\n")
            
            f.write("All class confidences:\n")
            for class_name, conf in sorted(result['all_confidences'].items(), 
                                          key=lambda x: x[1], reverse=True):
                f.write(f"  {class_name}: {conf:.4f}\n")
            f.write("\n")
        
        f.write("=" * 50 + "\n")

def write_results_to_json(results: List[Dict], output_path: str, frame_numbers: Optional[List[int]] = None, annotation_indices: Optional[List[str]] = None):
    """
    Write prediction results to a JSON file in a format optimized for processing.
    
    Args:
        results: List of prediction result dictionaries
        output_path: Path to save the output JSON file
        frame_numbers: Optional list of frame numbers corresponding to each result.
                       If not provided, will use indices as frame numbers.
        annotation_indices: Optional list of annotation indices corresponding to each result.
                           If not provided, will use indices as annotation indices.
    """
    import json
    import os
    import re
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create structured data for JSON format
    json_data = {}
    
    # Use provided frame numbers or default to indices
    if frame_numbers is None:
        frame_numbers = list(range(len(results)))
    
    # Use provided annotation indices or default to indices as strings
    if annotation_indices is None:
        annotation_indices = [str(i) for i in range(len(results))]
    
    # Ensure we have the right number of frame numbers and annotation indices
    if len(frame_numbers) != len(results) or len(annotation_indices) != len(results):
        raise ValueError(f"Number of frame numbers ({len(frame_numbers)}) or annotation indices ({len(annotation_indices)}) doesn't match number of results ({len(results)})")
    
    # Build the JSON structure: frame_number -> class_name -> {annotation_idx: confidence}
    for idx, (result, frame_num, anno_idx) in enumerate(zip(results, frame_numbers, annotation_indices)):
        # Convert frame number to string for JSON
        frame_key = str(frame_num)
        
        # Initialize frame entry if it doesn't exist
        if frame_key not in json_data:
            json_data[frame_key] = {}
        
        # Get all confidences for this result
        class_confidences = result['all_confidences']
        
        # For each class and its confidence
        for class_name, confidence in class_confidences.items():
            # Initialize class entry if it doesn't exist
            if class_name not in json_data[frame_key]:
                json_data[frame_key][class_name] = {}
            
            # Add the annotation index and confidence
            json_data[frame_key][class_name][anno_idx] = confidence
    
    # Ensure the output_path has a .json extension
    if not output_path.endswith('.json'):
        output_path = os.path.splitext(output_path)[0] + '.json'
    
    # Write to JSON file
    with open(output_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Results written to JSON file: {output_path}")

# Example usage
def inference():
    # Example usage of the FewShotPredictor
    # Get the path to model_weights directory (sibling to models directory)
    model_weights_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        "models/few_shot_models/save"
    )
    checkpoint_path = os.path.join(
        model_weights_dir, 
        "meta_mini-imagenet-1shot_meta-baseline-resnet12-max-va.pth"
    )
    
    predictor = FewShotPredictor(checkpoint_path=checkpoint_path)
    
    # Base directory where exported images are stored
    base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "datasets")
    support_base_dir = os.path.join(base_dir, "supports")
    query_dir = os.path.join(base_dir, "queries")
    
    # Check if directories exist
    if not os.path.exists(support_base_dir):
        print(f"Support directory not found at {support_base_dir}")
        exit(1)
    if not os.path.exists(query_dir):
        print(f"Query directory not found at {query_dir}")
        exit(1)
    
    # Dynamically load support class directories
    support_paths = {}
    for class_dir in os.listdir(support_base_dir):
        if os.path.isdir(os.path.join(support_base_dir, class_dir)):
            # Extract class name from directory name (format: class_id_class_name)
            try:
                class_parts = class_dir.split('_', 1)
                if len(class_parts) > 1:
                    class_name = class_parts[1]
                else:
                    class_name = class_dir
                    
                # Get all images for this class
                class_path = os.path.join(support_base_dir, class_dir)
                images = [os.path.join(class_path, f) for f in os.listdir(class_path) 
                         if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                
                if images:
                    support_paths[class_name] = images
                    print(f"Found {len(images)} support images for class '{class_name}'")
            except Exception as e:
                print(f"Error processing class directory {class_dir}: {e}")
    
    if not support_paths:
        print("No support images found in the directories")
        exit(1)
    
    # Get all query images
    query_paths = [os.path.join(query_dir, f) for f in os.listdir(query_dir)
                  if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    query_paths = []
    frame_numbers = []
    annotation_indices = []
    
    for f in os.listdir(query_dir):  # Changed from self.query_dir to query_dir
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            query_path = os.path.join(query_dir, f)  # Changed from self.query_dir to query_dir
            
            # Extract frame number and annotation index from filename
            # Expected format: framenumber_annotationidx.extension
            filename = os.path.splitext(f)[0]  # Remove extension
            parts = filename.split('_')
            
            try:
                if len(parts) >= 2:
                    # Keep frame number as string to preserve leading zeros
                    frame_num = parts[0]
                    
                    # For annotation index, convert to int first to remove leading zeros, then back to string
                    anno_idx = str(int(parts[1]))
                else:
                    # If format doesn't match, use default values
                    frame_num = str(len(query_paths))
                    anno_idx = "0"
                
                query_paths.append(query_path)
                frame_numbers.append(frame_num)
                annotation_indices.append(anno_idx)
            except ValueError:
                # If conversion fails, use default values
                print(f"Invalid filename format for {f}. Expected 'framenumber_annotationidx'")  # Changed from self.trainError.emit
                query_paths.append(query_path)
                frame_numbers.append(str(len(query_paths) - 1))
                annotation_indices.append("0")
    
    if not query_paths:
        print("No query images found")
        exit(1)
    
    print(f"Found {len(query_paths)} query images")
    
    # Create output directory for results
    output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
                             "results")
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Get predictions without fine-tuning
    results = predictor.predict(support_paths, query_paths)
    
    # Write results to file
    output_path = os.path.join(output_dir, f"predictions_no_finetune_{timestamp}.json")
    write_results_to_json(results, output_path, frame_numbers, annotation_indices)
    #write_results_to_file(results, output_path)
    print(f"Results without fine-tuning saved to: {output_path}")

    return output_path
    
    """ # Get predictions with fine-tuning
    results_finetuned = predictor.predict(support_paths, query_paths, finetune=True, finetune_steps=20, finetune_lr=0.01)
    
    # Write fine-tuned results to file
    output_path_ft = os.path.join(output_dir, f"predictions_with_finetune_{timestamp}.json")
    write_results_to_json(results_finetuned, output_path_ft)
    #write_results_to_file(results_finetuned, output_path_ft)
    print(f"Results with fine-tuning saved to: {output_path_ft}") """
