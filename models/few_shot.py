import numpy as np
import cv2
import torch
import os
import logging
""" from models.few_shot_models.siamese import SiameseModel
from models.few_shot_models.prototypical import PrototypicalModel
from models.few_shot_models.matching import MatchingModel
from models.few_shot_models.maml import MAMLModel
from models.few_shot_models.relation_network import RelationNetworkModel """
from utils.ensemble import ensemble_vote
from utils.data_augmentation import augment_image

# Set up logging
logger = logging.getLogger(__name__)

class FewShotEnsemble:
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        # Instantiate five different few-shot models.
        """ try:
            self.models = [
                SiameseModel(),
                PrototypicalModel(),
                MatchingModel(),
                MAMLModel(),
                RelationNetworkModel()
            ]
            logger.info("Successfully initialized all few-shot models")
        except Exception as e:
            logger.error(f"Error initializing few-shot models: {e}")
            self.models = [] """
            
        self.support_examples = {}  # {class_name: [image_patches]}
        self.augmentation_config = {
            "rotation_range": 15,
            "horizontal_flip": True,
            "brightness_limit": 0.2
        }

    def add_support_examples(self, class_name, examples):
        """Add support examples for a class"""
        if class_name not in self.support_examples:
            self.support_examples[class_name] = []
        
        self.support_examples[class_name].extend(examples)
        logger.info(f"Added {len(examples)} examples for class '{class_name}'")
        
        # Apply data augmentation to increase sample diversity
        augmented_examples = []
        for img in examples:
            # Generate 3 augmentations per original image
            for _ in range(3):
                aug_img = augment_image(
                    img, 
                    rotation_range=self.augmentation_config["rotation_range"],
                    horizontal_flip=self.augmentation_config["horizontal_flip"],
                    brightness_limit=self.augmentation_config["brightness_limit"]
                )
                augmented_examples.append(aug_img)
        
        self.support_examples[class_name].extend(augmented_examples)
        logger.info(f"Added {len(augmented_examples)} augmented examples for class '{class_name}'")
    
    def train(self):
        """Train all models using the support examples"""
        if not self.support_examples:
            logger.warning("No support examples available for training")
            return False
            
        """ try:
            for model in self.models:
                model.train(self.support_examples)
            logger.info("Successfully trained all models")
            return True
        except Exception as e:
            logger.error(f"Error during model training: {e}")
            return False
     """
    def predict(self, image_patch):
        """Make predictions using all trained models and ensemble the results"""
        if not self.models:
            logger.error("No models available for prediction")
            return "unknown", 0.0
            
        """ try:
            # Get predictions from each model
            predictions = []
            confidences = []
            
            for model in self.models:
                pred_class, conf = model.predict(image_patch)
                predictions.append(pred_class)
                confidences.append(conf)
                
            # Use ensemble voting to get final prediction
            final_class, final_conf = ensemble_vote(
                predictions, 
                confidences,
                threshold=self.confidence_threshold
            )
            
            logger.info(f"Ensemble prediction: {final_class} with confidence {final_conf:.4f}")
            return final_class, final_conf
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return "unknown", 0.0 """

if __name__ == "__main__":
    import sys
    import argparse
    
    # Configure logging for the script
    logging.basicConfig(level=logging.INFO, 
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Test Few-Shot Ensemble")
    parser.add_argument('--image', type=str, help='Path to test image')
    parser.add_argument('--support', type=str, help='Directory with support examples organized by class folders')
    args = parser.parse_args()
    
    ensemble = FewShotEnsemble()
    
    # Load support examples if provided
    if args.support and os.path.isdir(args.support):
        print(f"Loading support examples from {args.support}")
        
        for class_name in os.listdir(args.support):
            class_dir = os.path.join(args.support, class_name)
            if os.path.isdir(class_dir):
                examples = []
                for file in os.listdir(class_dir):
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        try:
                            img_path = os.path.join(class_dir, file)
                            img = cv2.imread(img_path)
                            if img is not None:
                                examples.append(img)
                        except Exception as e:
                            print(f"Error loading {file}: {e}")
                
                if examples:
                    print(f"Adding {len(examples)} support examples for class '{class_name}'")
                    ensemble.add_support_examples(class_name, examples)
        
        ensemble.train()
    
    # Test with provided image or exit
    if args.image and os.path.exists(args.image):
        print(f"Testing with image: {args.image}")
        test_image = cv2.imread(args.image)
        
        if test_image is not None:
            cls, conf = ensemble.predict(test_image)
            print(f"Prediction: {cls}, Confidence: {conf:.4f}")
        else:
            print(f"Failed to load test image: {args.image}")
    elif not args.support:
        print("Please provide either a test image or support examples directory.")
        print("Usage examples:")
        print("  python -m models.few_shot --image path/to/test_image.jpg")
        print("  python -m models.few_shot --support path/to/support_examples")
        print("  python -m models.few_shot --support path/to/support_examples --image path/to/test_image.jpg")
