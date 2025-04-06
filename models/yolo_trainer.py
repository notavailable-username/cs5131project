from ultralytics import YOLO
import os
import yaml

class YOLOTrainer:
    """YOLO model trainer with support for YOLOv8 and YOLOv11"""
    
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
        self.model_type = config.get("model_type", "YOLOv11")
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 16)
        self.img_size = config.get("img_size", 640)
        self.patience = config.get("patience", 20)
        self.use_augmentation = config.get("augmentation", True)
        self.preserve_aspect_ratio = config.get("preserve_aspect_ratio", True)
        self.dataset_yaml = config.get("dataset_yaml", None)
        self.output_dir = config.get("output_dir", None)
        
        # Get appropriate model path
        if self.model_type.startswith("YOLOv11"):
            self.model_path = "yolov11n"  # Use YOLOv11 nano by default
            if "YOLOv11s" in self.model_type:
                self.model_path = "yolov11s"
            elif "YOLOv11m" in self.model_type:
                self.model_path = "yolov11m"
            elif "YOLOv11l" in self.model_type:
                self.model_path = "yolov11l"
            elif "YOLOv11x" in self.model_type:
                self.model_path = "yolov11x"
        else:
            # Handle YOLOv8 variants
            self.model_path = self.model_type.lower()
    
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
                'device': 'cpu',  # Can be changed to 'cuda' for GPU training
                'project': os.path.dirname(self.output_dir) if self.output_dir else None,
                'name': os.path.basename(self.output_dir) if self.output_dir else None,
                'exist_ok': True,
            }
            
            # Configure aspect ratio preservation
            if self.preserve_aspect_ratio:
                train_args['rect'] = True  # Use rectangular training with aspect ratio preservation
            
            # Configure data augmentation
            if not self.use_augmentation:
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
    # Example usage
    config = {
        "model_type": "YOLOv11",
        "epochs": 10,
        "batch_size": 8,
        "img_size": 640,
        "patience": 15,
        "augmentation": True,
        "preserve_aspect_ratio": True,
        "dataset_yaml": "path/to/dataset.yaml",
        "output_dir": "path/to/output"
    }
    trainer = YOLOTrainer(config)
    results = trainer.train()
    print(results)
