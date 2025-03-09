import argparse
import os
import yaml
import cv2
from models.yolo_trainer import YOLOTrainer
from models.few_shot import FewShotEnsemble
from models.motion_detector import MotionDetector

def run_cli(config):
    parser = argparse.ArgumentParser(description="Command Line Interface for Model Training")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # YOLO training sub-command
    yolo_parser = subparsers.add_parser("train_yolo", help="Train YOLOv11")
    yolo_parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")

    # Few-shot training sub-command
    few_shot_parser = subparsers.add_parser("train_few_shot", help="Train few-shot models")
    few_shot_parser.add_argument("--support-dir", required=True, help="Directory containing support images organized in class subfolders")
    few_shot_parser.add_argument("--confidence", type=float, default=0.5, help="Confidence threshold (default: 0.5)")
    few_shot_parser.add_argument("--output-dir", help="Directory to save trained models (optional)")

    # Motion detector sub-command
    motion_parser = subparsers.add_parser("motion_detection", help="Run motion detection on a video")
    motion_parser.add_argument("--video", required=True, help="Path to the video file")
    motion_parser.add_argument("--output-dir", required=True, help="Directory to save detection results")
    motion_parser.add_argument("--sensitivity", choices=["low", "medium", "high"], default="medium", 
                             help="Sensitivity presets (low: 25, medium: 16, high: 10)")
    motion_parser.add_argument("--display", action="store_true", help="Display detection results in real-time")

    args = parser.parse_args()

    if args.command == "train_yolo":
        trainer = YOLOTrainer(config["yolo"])
        trainer.train(use_pretrained=args.pretrained)
    
    elif args.command == "train_few_shot":
        if not os.path.isdir(args.support_dir):
            print(f"Error: Support directory '{args.support_dir}' not found")
            return
        
        # Load support examples from directory structure
        # Expecting folders named by class containing support images
        support_examples = {}
        for class_name in os.listdir(args.support_dir):
            class_dir = os.path.join(args.support_dir, class_name)
            if os.path.isdir(class_dir):
                support_examples[class_name] = []
                for img_file in os.listdir(class_dir):
                    if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(class_dir, img_file)
                        img = cv2.imread(img_path)
                        if img is not None:
                            support_examples[class_name].append(img)
                print(f"Loaded {len(support_examples[class_name])} support examples for class '{class_name}'")
        
        # Initialize and train few-shot ensemble
        few_shot = FewShotEnsemble(confidence_threshold=args.confidence)
        
        # Add support examples
        for class_name, examples in support_examples.items():
            few_shot.add_support_examples(class_name, examples)
        
        # Train models
        if few_shot.train():
            print("Few-shot models trained successfully")
            
        # TODO: Add model saving logic if output_dir is provided
    
    elif args.command == "motion_detection":
        if not os.path.exists(args.video):
            print(f"Error: Video file '{args.video}' not found")
            return
        
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        
        # Configure motion detector based on sensitivity
        var_threshold = 16  # default medium
        if args.sensitivity == "low":
            var_threshold = 25
        elif args.sensitivity == "high":
            var_threshold = 10
        
        detector = MotionDetector(varThreshold=var_threshold)
        
        # Process video
        cap = cv2.VideoCapture(args.video)
        frame_count = 0
        detection_count = 0
        
        print(f"Processing video: {args.video}")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Detect motion
            boxes = detector.detect(frame)
            
            if boxes:
                detection_count += len(boxes)
                # Save frame with boxes
                result_frame = frame.copy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(result_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Save the frame with detections
                output_path = os.path.join(args.output_dir, f"detection_{frame_count:04d}.jpg")
                cv2.imwrite(output_path, result_frame)
            
            if args.display:
                # Display the frame with boxes
                display_frame = frame.copy()
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.imshow("Motion Detection", display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, found {detection_count} detections")
        
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        
        print(f"Motion detection complete. Processed {frame_count} frames and saved {detection_count} detections to {args.output_dir}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    # Load config if running as standalone script
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    
    config = {}
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"Warning: Config file not found at {config_path}, using default settings")
    
    run_cli(config)
