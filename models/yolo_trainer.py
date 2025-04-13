from ultralytics import YOLO
import os
import yaml
import numpy as np

class YOLOTrainer:
    """YOLO model trainer with support for YOLOv8, YOLOv9, YOLOv10 and YOLOv11"""
    
    def __init__(self, config):
        """
        Initialize the YOLO trainer with configuration.
        
        Args:
            config (dict): Training configuration with following keys:
                - model_type: Type of YOLO model (YOLOv8n, YOLOv11, etc.)
                - epochs: Number of training epochs
                - batch_size: Batch size for training
                - img_size: Input image size
                - patience: Early stopping patience
                - augmentation: Whether to use data augmentation
                - preserve_aspect_ratio: Whether to preserve aspect ratio during resizing
                - dataset_yaml: Path to dataset.yaml
                - output_dir: Directory to save training outputs
        """
        self.model_type = config.get("model_type", "YOLOv11n")
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 16)
        self.img_size = config.get("img_size", 640)
        self.patience = config.get("patience", 20)
        self.use_augmentation = config.get("augmentation", True)
        self.preserve_aspect_ratio = config.get("preserve_aspect_ratio", True)
        self.dataset_yaml = config.get("dataset_yaml", None)
        self.output_dir = config.get("output_dir", None)
        
        # Augmentation settings
        self.mosaic = config.get("mosaic", 1.0)
        self.mixup = config.get("mixup", 0.1)
        self.degrees = config.get("degrees", 0.0)
        self.translate = config.get("translate", 0.1)
        self.scale = config.get("scale", 0.5)
        self.shear = config.get("shear", 0.0)
        self.perspective = config.get("perspective", 0.0)
        self.flipud = config.get("flipud", 0.0)
        self.fliplr = config.get("fliplr", 0.5)
        self.hsv_h = config.get("hsv_h", 0.015)
        self.hsv_s = config.get("hsv_s", 0.7)
        self.hsv_v = config.get("hsv_v", 0.4)
        
        # Get appropriate model path
        if self.model_type.startswith("YOLOv11"):
            variant = self.model_type[-1].lower()  # Extract size variant (n, s, m, l, x)
            self.model_path = f"yolo11{variant}.pt"  # YOLO11 doesn't use 'v' in its path
        elif self.model_type.startswith("YOLOv10"):
            variant = self.model_type[-1].lower()  # Extract size variant
            self.model_path = f"yolov10{variant}.pt"
        elif self.model_type.startswith("YOLOv9"):
            variant = self.model_type[-1].lower()  # Extract size variant
            if variant == 'c':  # Special case for YOLOv9c
                self.model_path = "yolov9c.pt"
            elif variant == 'e':  # Special case for YOLOv9e
                self.model_path = "yolov9e.pt"
            else:
                self.model_path = f"yolov9{variant}.pt"
        else:
            # Handle YOLOv8 variants
            self.model_path = f"{self.model_type.lower()}.pt"
    
    def train(self):
        """Train the YOLO model using the provided configuration"""
        try:
            # Initialize model
            model = YOLO(self.model_path)
            
            # Set training arguments
            train_args = {
                'data': self.dataset_yaml,
                'epochs': self.epochs,
                'batch': self.batch_size,
                'imgsz': self.img_size,
                'patience': self.patience,
                'device': 'cuda',  # Can be changed to 'cuda' for GPU training
                'project': os.path.dirname(self.output_dir) if self.output_dir else None,
                'name': os.path.basename(self.output_dir) if self.output_dir else None,
                'exist_ok': True,
                'verbose': True,  # Enable verbose output for better logging
            }
            
            # Configure aspect ratio preservation
            if self.preserve_aspect_ratio:
                train_args['rect'] = True  # Use rectangular training with aspect ratio preservation
            
            # Configure data augmentation
            if self.use_augmentation:
                # Set all augmentation parameters from config
                train_args['mosaic'] = self.mosaic
                train_args['mixup'] = self.mixup
                train_args['degrees'] = self.degrees
                train_args['translate'] = self.translate
                train_args['scale'] = self.scale
                train_args['shear'] = self.shear
                train_args['perspective'] = self.perspective
                train_args['flipud'] = self.flipud
                train_args['fliplr'] = self.fliplr
                train_args['hsv_h'] = self.hsv_h
                train_args['hsv_s'] = self.hsv_s
                train_args['hsv_v'] = self.hsv_v
            else:
                # Disable augmentation techniques
                train_args['augment'] = False
                train_args['mosaic'] = 0.0
                train_args['mixup'] = 0.0
                train_args['degrees'] = 0.0
                train_args['shear'] = 0.0
                train_args['perspective'] = 0.0
                train_args['flipud'] = 0.0
                train_args['fliplr'] = 0.0
                train_args['hsv_h'] = 0.0
                train_args['hsv_s'] = 0.0
                train_args['hsv_v'] = 0.0
                train_args['translate'] = 0.0
                train_args['scale'] = 0.0
            
            # Start training
            results = model.train(**train_args)
            
            # Return the results
            return {
                'success': True,
                'mAP': results.results_dict.get('metrics/mAP50-95(B)', 0),
                'model_path': os.path.join(self.output_dir, 'weights/best.pt')
            }
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

class TrainingThread:
    """Thread for running the training in the background"""
    def __init__(self, trainer, dataset_dir, output_dir, config):
        """
        Initialize the training thread.
        
        Args:
            trainer: YOLOTrainer instance
            dataset_dir: Directory containing the training data
            output_dir: Directory to save outputs
            config: Training configuration
        """
        self.trainer = trainer
        self.dataset_dir = dataset_dir
        self.output_dir = output_dir
        self.config = config
    
    def start(self):
        """Run the training process"""
        try:
            results = self.trainer.train()
            return results
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

if __name__ == "__main__":
    import argparse
    
    # Create argument parser
    parser = argparse.ArgumentParser(description="YOLO Model Trainer")
    
    # Add arguments for common training parameters
    parser.add_argument("--model", type=str, default="YOLOv11n", help="Model type (YOLOv8n, YOLOv9c, YOLOv10n, YOLOv11n, etc.)")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--img-size", type=int, default=640, help="Input image size")
    parser.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    parser.add_argument("--dataset", type=str, required=True, help="Path to dataset.yaml file")
    parser.add_argument("--output", type=str, required=True, help="Output directory for trained model")
    parser.add_argument("--no-augmentation", action="store_true", help="Disable data augmentation")
    parser.add_argument("--no-aspect-ratio", action="store_true", help="Disable aspect ratio preservation")
    
    # Add augmentation parameters
    parser.add_argument("--mosaic", type=float, default=1.0, help="Mosaic augmentation (0.0-1.0)")
    parser.add_argument("--mixup", type=float, default=0.1, help="Mixup augmentation (0.0-1.0)")
    parser.add_argument("--degrees", type=float, default=0.0, help="Rotation augmentation degrees")
    parser.add_argument("--translate", type=float, default=0.1, help="Translation augmentation")
    parser.add_argument("--scale", type=float, default=0.5, help="Scale augmentation")
    parser.add_argument("--shear", type=float, default=0.0, help="Shear augmentation")
    parser.add_argument("--perspective", type=float, default=0.0, help="Perspective augmentation")
    parser.add_argument("--flipud", type=float, default=0.0, help="Flip up/down probability")
    parser.add_argument("--fliplr", type=float, default=0.5, help="Flip left/right probability")
    parser.add_argument("--hsv-h", type=float, default=0.015, help="HSV hue augmentation")
    parser.add_argument("--hsv-s", type=float, default=0.7, help="HSV saturation augmentation")
    parser.add_argument("--hsv-v", type=float, default=0.4, help="HSV value augmentation")
    parser.add_argument("--device", type=str, default="", help="Device to run on (cpu, 0, 0,1,2,3, etc.)")
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = {
        "model_type": args.model,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "patience": args.patience,
        "augmentation": not args.no_augmentation,
        "preserve_aspect_ratio": not args.no_aspect_ratio,
        "dataset_yaml": args.dataset,
        "output_dir": args.output,
        
        # Augmentation settings
        "mosaic": args.mosaic,
        "mixup": args.mixup,
        "degrees": args.degrees,
        "translate": args.translate,
        "scale": args.scale,
        "shear": args.shear,
        "perspective": args.perspective,
        "flipud": args.flipud,
        "fliplr": args.fliplr,
        "hsv_h": args.hsv_h,
        "hsv_s": args.hsv_s,
        "hsv_v": args.hsv_v
    }
    
    # Add device if specified
    if args.device:
        config["device"] = args.device
    
    print(f"Starting YOLO training with {args.model} model...")
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {args.output}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, Image size: {args.img_size}")
    print(f"Augmentation enabled: {not args.no_augmentation}")
    
    # Create trainer and run training
    trainer = YOLOTrainer(config)
    results = trainer.train()
    
    # Print results
    if results["success"]:
        print(f"\nTraining completed successfully!")
        print(f"Model saved to: {results.get('model_path', 'unknown')}")
        print(f"mAP: {results.get('mAP', 0):.4f}")
    else:
        print(f"\nTraining failed: {results.get('error', 'unknown error')}")
