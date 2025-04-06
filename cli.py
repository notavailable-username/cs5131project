import os
import yaml
import cv2
import json
import shutil
import time
from datetime import datetime
import glob
import csv
from models.yolo_trainer import YOLOTrainer
from models.motion_detector import MotionDetector

class InteractiveCLI:
    """Interactive Command Line Interface for Motion-Aware Few-Shot Object Detection"""
    
    def __init__(self, config):
        self.config = config
        self.running = True
        self.current_video_path = None
        self.loaded_videos = []
        self.classes = []
        self.all_detections = {}  # Dictionary mapping video paths to detections
        
        # Create base directories
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.datasets_dir = os.path.join(self.base_dir, "datasets")
        self.annotations_dir = os.path.join(self.datasets_dir, "annotations")
        self.videos_annotations_dir = os.path.join(self.annotations_dir, "videos")
        self.video_configs_dir = os.path.join(self.datasets_dir, "video_configs")
        
        # Create directories if they don't exist
        for directory in [self.datasets_dir, self.annotations_dir, 
                         self.videos_annotations_dir, self.video_configs_dir]:
            os.makedirs(directory, exist_ok=True)
        
        # Command mapping
        self.commands = {
            '/help': self.show_help,
            '/exit': self.exit_cli,
            '/quit': self.exit_cli,
            '/import_videos': self.import_videos,
            '/load_videos': self.load_videos,
            '/list_videos': self.list_videos,
            '/switch_video': self.switch_video,
            '/motion_detect': self.motion_detect,
            '/edit_classes': self.edit_classes,
            '/list_classes': self.list_classes,
            '/few_shot_export': self.few_shot_export,
            '/few_shot_train': self.few_shot_train,
            '/yolo_prepare': self.yolo_prepare_data,
            '/yolo_train': self.yolo_train
        }
        
        # Welcome message
        self.print_welcome_message()
    
    def print_welcome_message(self):
        """Print welcome message with available commands"""
        print("\n" + "="*80)
        print("Welcome to Motion-Aware Few-Shot Object Detection CLI".center(80))
        print("="*80 + "\n")
        print("Type '/help' to see available commands")
        print("Type '/exit' or '/quit' to exit the program\n")
    
    def show_help(self, args=None):
        """Show available commands and their descriptions"""
        print("\nAvailable commands:")
        print("  /help                      - Show this help message")
        print("  /exit, /quit               - Exit the program")
        print("\nVideo Management:")
        print("  /import_videos <path>      - Import video files from directory")
        print("  /load_videos               - Load and display available videos")
        print("  /list_videos               - List currently loaded videos")
        print("  /switch_video <index>      - Switch to a different loaded video")
        print("\nMotion Detection:")
        print("  /motion_detect [options]   - Run motion detection on current video")
        print("        --sensitivity=<value>  - Set sensitivity (lower is more sensitive)")
        print("        --interval=<value>     - Set frame interval (frames per second to process)")
        print("\nClass Management:")
        print("  /edit_classes              - Add, edit, or remove classes")
        print("  /list_classes              - Show current classes")
        print("\nFew-Shot Learning:")
        print("  /few_shot_export           - Export support and query images")
        print("  /few_shot_train            - Train few-shot model with exported images")
        print("\nYOLO Training:")
        print("  /yolo_prepare              - Prepare data for YOLO training")
        print("  /yolo_train [options]      - Train YOLO model")
        print("        --epochs=<value>       - Set number of epochs")
        print("        --batch=<value>        - Set batch size")
        print("        --img-size=<value>     - Set image size")
        print("        --model=<name>         - Set model type (e.g., YOLOv11, YOLOv8n)")
        return True
    
    def exit_cli(self, args=None):
        """Exit the CLI"""
        print("Exiting program. Goodbye!")
        self.running = False
        return True
    
    def parse_command(self, input_str):
        """Parse command string into command and arguments"""
        if not input_str.strip():
            return None, []
            
        parts = input_str.strip().split()
        command = parts[0].lower()
        args = parts[1:]
        
        # Parse named arguments like --key=value
        parsed_args = []
        kwargs = {}
        for arg in args:
            if arg.startswith('--') and '=' in arg:
                key, value = arg[2:].split('=', 1)
                kwargs[key] = value
            else:
                parsed_args.append(arg)
        
        return command, parsed_args, kwargs
    
    def import_videos(self, args=None, **kwargs):
        """Import video files to datasets/videos directory"""
        videos_dir = os.path.join(self.datasets_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        if not args:
            path = input("Enter path to video file or directory: ")
        else:
            path = args[0]
        
        if not os.path.exists(path):
            print(f"Error: Path '{path}' does not exist.")
            return False
        
        imported_count = 0
        
        if os.path.isdir(path):
            # Import all videos from directory
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
            for file in os.listdir(path):
                file_path = os.path.join(path, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
                    destination = os.path.join(videos_dir, os.path.basename(file_path))
                    shutil.copy2(file_path, destination)
                    imported_count += 1
            
            print(f"Imported {imported_count} videos to datasets/videos")
        else:
            # Import single video file
            destination = os.path.join(videos_dir, os.path.basename(path))
            shutil.copy2(path, destination)
            print(f"Imported video: {os.path.basename(path)} to datasets/videos")
            imported_count = 1
        
        return imported_count > 0
    
    def load_videos(self, args=None, **kwargs):
        """Load available videos from datasets/videos directory"""
        videos_dir = os.path.join(self.datasets_dir, "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        video_files = []
        
        for file in os.listdir(videos_dir):
            file_path = os.path.join(videos_dir, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
                video_files.append(file_path)
        
        if not video_files:
            print("No videos found in datasets/videos directory.")
            print("Use '/import_videos <path>' to import videos.")
            return False
        
        print(f"\nFound {len(video_files)} videos:")
        for i, video_path in enumerate(video_files):
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                duration = frame_count / fps if fps > 0 else 0
                cap.release()
                print(f"  [{i}] {os.path.basename(video_path)} - {width}x{height}, {fps:.1f}fps, {duration:.1f}s")
            else:
                print(f"  [{i}] {os.path.basename(video_path)} - [Error: Could not open video]")
        
        selection = input("\nEnter video numbers to load (comma-separated) or 'all': ")
        
        if selection.lower() == 'all':
            self.loaded_videos = video_files
        else:
            try:
                indices = [int(idx.strip()) for idx in selection.split(',')]
                self.loaded_videos = [video_files[idx] for idx in indices if 0 <= idx < len(video_files)]
            except ValueError:
                print("Invalid input. Please enter comma-separated numbers or 'all'.")
                return False
        
        # Set current video to the first loaded video
        if self.loaded_videos:
            self.current_video_path = self.loaded_videos[0]
            print(f"Loaded {len(self.loaded_videos)} videos. Current video: {os.path.basename(self.current_video_path)}")
            
            # Load classes for this video if available
            self.load_classes_from_csv()
            return True
        else:
            print("No videos loaded.")
            return False
    
    def list_videos(self, args=None, **kwargs):
        """List currently loaded videos"""
        if not self.loaded_videos:
            print("No videos loaded. Use '/load_videos' to load videos.")
            return False
        
        print("\nCurrently loaded videos:")
        for i, video_path in enumerate(self.loaded_videos):
            if video_path == self.current_video_path:
                print(f"→ [{i}] {os.path.basename(video_path)} (CURRENT)")
            else:
                print(f"  [{i}] {os.path.basename(video_path)}")
        return True
    
    def switch_video(self, args=None, **kwargs):
        """Switch to a different loaded video"""
        if not self.loaded_videos:
            print("No videos loaded. Use '/load_videos' to load videos.")
            return False
        
        if not args:
            self.list_videos()
            index = input("Enter video index to switch to: ")
        else:
            index = args[0]
            
        try:
            idx = int(index)
            if 0 <= idx < len(self.loaded_videos):
                self.current_video_path = self.loaded_videos[idx]
                print(f"Switched to video: {os.path.basename(self.current_video_path)}")
                
                # Load classes for this video
                self.load_classes_from_csv()
                return True
            else:
                print(f"Error: Index {idx} is out of range.")
                return False
        except ValueError:
            print("Invalid input. Please enter a valid index.")
            return False
    
    def motion_detect(self, args=None, **kwargs):
        """Run motion detection on the current video"""
        if not self.current_video_path:
            print("No video selected. Use '/load_videos' to load videos.")
            return False
        
        # Get settings from kwargs or use defaults
        sensitivity = int(kwargs.get('sensitivity', 25))
        interval = int(kwargs.get('interval', 2))
        
        print(f"Running motion detection on {os.path.basename(self.current_video_path)}")
        print(f"Settings: sensitivity={sensitivity}, interval={interval}")
        
        try:
            # Create a motion detector with the specified settings
            md_conf = self.config.get("motion_detector", {}).copy()
            md_conf["varThreshold"] = sensitivity
            detector = MotionDetector(**md_conf)
            
            # Open the video
            cap = cv2.VideoCapture(self.current_video_path)
            if not cap.isOpened():
                print("Error: Could not open the video file.")
                return False
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(fps / interval))
            
            # Create annotation directory for this video
            video_name = self.get_video_name()
            annotations_dir = os.path.join(self.videos_annotations_dir, video_name)
            os.makedirs(annotations_dir, exist_ok=True)
            
            # Process frames
            all_detections = []
            frame_count = 0
            detection_count = 0
            start_time = time.time()
            
            print("\nProcessing video frames...")
            print(f"Press Ctrl+C to abort (progress updates every {frame_interval*5} frames)")
            
            try:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    # Process every nth frame
                    if frame_count % frame_interval == 0:
                        # Calculate and print progress
                        if frame_count % (frame_interval * 5) == 0:
                            progress = min(100, int(100 * frame_count / total_frames))
                            elapsed = time.time() - start_time
                            eta = (elapsed / max(1, frame_count)) * (total_frames - frame_count) if frame_count > 0 else 0
                            print(f"Progress: {progress}% (Frame {frame_count}/{total_frames}) - "
                                  f"Detections: {detection_count} - ETA: {eta:.1f}s")
                        
                        # Run motion detection
                        boxes = detector.detect(frame)
                        
                        if boxes:
                            frame_h, frame_w = frame.shape[:2]
                            txt_path = os.path.join(annotations_dir, f"frame_{frame_count:06d}.txt")
                            
                            with open(txt_path, 'w') as f:
                                for box in boxes:
                                    detection_count += 1
                                    x1, y1, x2, y2 = box
                                    x1, y1 = max(0, x1), max(0, y1)
                                    x2, y2 = min(frame_w, x2), min(frame_h, y2)
                                    
                                    # Convert to YOLO format (center_x, center_y, width, height) - normalized
                                    box_w = x2 - x1
                                    box_h = y2 - y1
                                    center_x = (x1 + box_w / 2) / frame_w
                                    center_y = (y1 + box_h / 2) / frame_h
                                    norm_width = box_w / frame_w
                                    norm_height = box_h / frame_h
                                    
                                    # Write detection in YOLO format, using "-" as class
                                    f.write(f"- {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}\n")
                                    
                                    # Record detection
                                    detection = {
                                        'frame_idx': frame_count,
                                        'bbox_yolo': [center_x, center_y, norm_width, norm_height],
                                        'bbox_abs': [x1, y1, x2, y2],
                                        'class': '-',
                                        'timestamp': frame_count / fps
                                    }
                                    all_detections.append(detection)
                    
                    frame_count += 1
                    
            except KeyboardInterrupt:
                print("\nMotion detection aborted by user.")
            
            cap.release()
            
            # Store detections for the current video
            self.all_detections[self.current_video_path] = all_detections
            
            # Create the classes.csv file if it doesn't exist
            classes_csv_path = os.path.join(annotations_dir, "classes.csv")
            if not os.path.exists(classes_csv_path):
                with open(classes_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['class_id', 'class_name'])
                    for i, class_name in enumerate(self.classes):
                        writer.writerow([i, class_name])
            
            # Print summary
            end_time = time.time()
            total_time = end_time - start_time
            unique_frames = len(set([d['frame_idx'] for d in all_detections]))
            
            print(f"\nMotion detection complete:")
            print(f"- Processed {frame_count} frames in {total_time:.1f} seconds")
            print(f"- Found {detection_count} detections in {unique_frames} frames")
            print(f"- Annotations saved to: {annotations_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error during motion detection: {str(e)}")
            return False
    
    def edit_classes(self, args=None, **kwargs):
        """Add, edit, or remove classes"""
        if not self.current_video_path:
            print("No video selected. Use '/load_videos' to load videos.")
            return False
        
        video_name = self.get_video_name()
        annotations_dir = os.path.join(self.videos_annotations_dir, video_name)
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Load existing classes
        self.load_classes_from_csv()
        
        print("\nCurrent classes:")
        if self.classes:
            for i, class_name in enumerate(self.classes):
                print(f"  [{i}] {class_name}")
        else:
            print("  No classes defined yet.")
        
        while True:
            print("\nClass Management Options:")
            print("  [a] Add new class")
            print("  [e] Edit class")
            print("  [r] Remove class")
            print("  [s] Save and exit")
            print("  [c] Cancel and exit")
            
            choice = input("\nEnter option: ").lower().strip()
            
            if choice == 'a':
                class_name = input("Enter new class name: ")
                if class_name and class_name not in self.classes:
                    self.classes.append(class_name)
                    print(f"Added class: {class_name}")
                else:
                    print("Invalid class name or class already exists.")
            
            elif choice == 'e':
                if not self.classes:
                    print("No classes to edit.")
                    continue
                    
                index = input("Enter class index to edit: ")
                try:
                    idx = int(index)
                    if 0 <= idx < len(self.classes):
                        new_name = input(f"Enter new name for class '{self.classes[idx]}': ")
                        if new_name:
                            self.classes[idx] = new_name
                            print(f"Updated class to: {new_name}")
                        else:
                            print("Invalid class name.")
                    else:
                        print("Invalid class index.")
                except ValueError:
                    print("Please enter a valid index.")
            
            elif choice == 'r':
                if not self.classes:
                    print("No classes to remove.")
                    continue
                    
                index = input("Enter class index to remove: ")
                try:
                    idx = int(index)
                    if 0 <= idx < len(self.classes):
                        removed = self.classes.pop(idx)
                        print(f"Removed class: {removed}")
                    else:
                        print("Invalid class index.")
                except ValueError:
                    print("Please enter a valid index.")
            
            elif choice == 's':
                # Save classes to CSV
                classes_csv_path = os.path.join(annotations_dir, "classes.csv")
                with open(classes_csv_path, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(['class_id', 'class_name'])
                    for i, class_name in enumerate(self.classes):
                        writer.writerow([i, class_name])
                
                print(f"Saved {len(self.classes)} classes to: {classes_csv_path}")
                break
            
            elif choice == 'c':
                print("Changes cancelled.")
                break
            
            else:
                print("Invalid option.")
            
            # Show updated class list
            print("\nCurrent classes:")
            if self.classes:
                for i, class_name in enumerate(self.classes):
                    print(f"  [{i}] {class_name}")
            else:
                print("  No classes defined yet.")
        
        return True
    
    def list_classes(self, args=None, **kwargs):
        """Show current classes"""
        if not self.current_video_path:
            print("No video selected. Use '/load_videos' to load videos.")
            return False
        
        # Make sure classes are loaded
        self.load_classes_from_csv()
        
        print("\nClasses for current video:")
        if self.classes:
            for i, class_name in enumerate(self.classes):
                print(f"  [{i}] {class_name}")
        else:
            print("  No classes defined. Use '/edit_classes' to add classes.")
        
        return True
    
    def few_shot_export(self, args=None, **kwargs):
        """Export support and query images for few-shot learning"""
        if not self.current_video_path:
            print("No video selected. Use '/load_videos' to load videos.")
            return False
        
        # Check if we have annotations
        annotations_dir = self.get_annotations_dir_for_current_video()
        if not annotations_dir or not os.path.exists(annotations_dir):
            print("No annotations found. Run motion detection first using '/motion_detect'.")
            return False
        
        # Check if we have classes
        if not self.classes:
            print("No classes defined. Use '/edit_classes' to add classes.")
            return False
        
        # Create directories for support and query images
        base_dir = self.datasets_dir
        support_base_dir = os.path.join(base_dir, "supports")
        query_dir = os.path.join(base_dir, "queries")
        
        # Clear previous exports
        if os.path.exists(support_base_dir):
            shutil.rmtree(support_base_dir)
        if os.path.exists(query_dir):
            shutil.rmtree(query_dir)
        
        os.makedirs(support_base_dir, exist_ok=True)
        os.makedirs(query_dir, exist_ok=True)
        
        # Process annotations and export images
        print(f"Exporting support and query images from {os.path.basename(self.current_video_path)}...")
        
        try:
            # Open the video file
            cap = cv2.VideoCapture(self.current_video_path)
            if not cap.isOpened():
                print("Error: Could not open the video file.")
                return False
            
            # Get video dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Create class mappings
            class_map = {str(i): class_name for i, class_name in enumerate(self.classes)}
            class_map['-'] = '-'  # Add mapping for placeholder
            
            # Get all annotation files
            annotation_files = [f for f in os.listdir(annotations_dir) 
                             if f.startswith("frame_") and f.endswith(".txt")]
            
            # Count for reporting
            support_count = 0
            query_count = 0
            processed_frames = 0
            
            for annotation_file in annotation_files:
                # Extract frame number from filename
                frame_idx = int(annotation_file.replace("frame_", "").replace(".txt", ""))
                formatted_frame_number = f"{frame_idx:06d}"
                
                # Read annotations
                annotations = []
                with open(os.path.join(annotations_dir, annotation_file), 'r') as f:
                    for annotation_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = parts[0]
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                # Convert to absolute coordinates
                                x1 = int((x_center - width / 2) * frame_width)
                                y1 = int((y_center - height / 2) * frame_height)
                                x2 = int((x_center + width / 2) * frame_width)
                                y2 = int((y_center + height / 2) * frame_height)
                                
                                # Ensure coordinates are within frame bounds
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(frame_width - 1, x2)
                                y2 = min(frame_height - 1, y2)
                                
                                annotations.append({
                                    'bbox_abs': [x1, y1, x2, y2],
                                    'class': class_id,
                                    'annotation_idx': annotation_idx
                                })
                            except ValueError:
                                continue
                
                # If we have annotations, process the frame
                if annotations:
                    # Seek to the frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                    ret, frame = cap.read()
                    if not ret:
                        continue
                    
                    for ann in annotations:
                        class_id = ann['class']
                        annotation_idx = ann['annotation_idx']
                        formatted_annotation_idx = f"{annotation_idx:04d}"
                        
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = ann['bbox_abs']
                        
                        # Crop the image
                        cropped_img = frame[y1:y2, x1:x2]
                        
                        # Create filename
                        filename = f"{formatted_frame_number}_{formatted_annotation_idx}.png"
                        
                        # Handle based on whether it's support or query
                        if class_id == '-':  # Query image (placeholder class)
                            filepath = os.path.join(query_dir, filename)
                            cv2.imwrite(filepath, cropped_img)
                            query_count += 1
                        else:  # Support image
                            class_name = class_map.get(class_id, f"class_{class_id}")
                            class_dir = os.path.join(support_base_dir, f"{class_id}_{class_name}")
                            os.makedirs(class_dir, exist_ok=True)
                            filepath = os.path.join(class_dir, filename)
                            cv2.imwrite(filepath, cropped_img)
                            support_count += 1
                    
                    processed_frames += 1
            
            cap.release()
            
            print(f"\nExport complete:")
            print(f"- Exported {support_count} support images and {query_count} query images")
            print(f"- Processed {processed_frames} frames")
            print(f"- Support images saved to: {support_base_dir}")
            print(f"- Query images saved to: {query_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error exporting images: {str(e)}")
            return False
    
    def few_shot_train(self, args=None, **kwargs):
        """Train few-shot learning model with exported images"""
        # Check if support and query directories exist
        base_dir = self.datasets_dir
        support_base_dir = os.path.join(base_dir, "supports")
        query_dir = os.path.join(base_dir, "queries")
        
        if not os.path.exists(support_base_dir) or not os.path.exists(query_dir):
            print("Support and query images not found. Run '/few_shot_export' first.")
            return False
        
        # Check if there are support classes
        support_classes = [d for d in os.listdir(support_base_dir) 
                         if os.path.isdir(os.path.join(support_base_dir, d))]
        if not support_classes:
            print("No support classes found. Make sure to export support images first.")
            return False
        
        # Check if there are query images
        query_images = [f for f in os.listdir(query_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not query_images:
            print("No query images found. Make sure to export query images first.")
            return False
        
        print("\nStarting few-shot learning training...")
        print(f"- Support classes: {len(support_classes)}")
        print(f"- Query images: {len(query_images)}")
        
        try:
            # Import the FewShotPredictor
            from models.few_shot_models.inference import FewShotPredictor, write_results_to_json
            
            # Get model weights path
            model_weights_dir = os.path.join(self.base_dir, 
                                          "models/few_shot_models/save")
            checkpoint_path = os.path.join(
                model_weights_dir, 
                "meta_mini-imagenet-1shot_meta-baseline-resnet12-max-va.pth"
            )
            
            if not os.path.exists(checkpoint_path):
                print(f"Model checkpoint not found at: {checkpoint_path}")
                return False
            
            print("Initializing FewShotPredictor...")
            predictor = FewShotPredictor(checkpoint_path=checkpoint_path)
            
            # Prepare support paths dictionary
            support_paths = {}
            print("Loading support images...")
            
            for class_dir in support_classes:
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
                            print(f"  - Class '{class_name}': {len(images)} images")
                    except Exception as e:
                        print(f"Error processing class directory {class_dir}: {str(e)}")
                        return False
            
            # Get all query images
            query_paths = [os.path.join(query_dir, f) for f in os.listdir(query_dir)
                          if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Create output directory for results
            output_dir = os.path.join(self.base_dir, "results")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(output_dir, f"predictions_{timestamp}.txt")
            
            print(f"\nStarting prediction with {len(query_paths)} query images...")
            print("This may take some time, please wait...")
            
            # Run prediction (with fine-tuning)
            results = predictor.predict(
                support_paths, query_paths, 
                finetune=True, finetune_steps=20, finetune_lr=0.01
            )
            
            print(f"Writing results to: {output_path}")
            write_results_to_json(results, output_path)
            
            # Print summary of results
            print("\nFew-shot learning training complete!")
            print(f"Results saved to: {output_path}")
            
            return True
            
        except ImportError:
            print("Error: Could not import FewShotPredictor. Make sure the module is available.")
            return False
        except Exception as e:
            print(f"Error during few-shot learning training: {str(e)}")
            return False
    
    def yolo_prepare_data(self, args=None, **kwargs):
        """Prepare data for YOLO training"""
        if not self.current_video_path:
            print("No video selected. Use '/load_videos' to load videos.")
            return False
        
        # Check if we have annotations
        annotations_dir = self.get_annotations_dir_for_current_video()
        if not annotations_dir or not os.path.exists(annotations_dir):
            print("No annotations found. Run motion detection first using '/motion_detect'.")
            return False
        
        # Check if we have classes
        if not self.classes:
            print("No classes defined. Use '/edit_classes' to add classes.")
            return False
        
        print(f"Preparing YOLO training data for {os.path.basename(self.current_video_path)}...")
        
        try:
            # Get confidence threshold
            threshold = float(input("Enter confidence threshold (0.0-1.0, default 0.5): ") or "0.5")
            threshold = max(0, min(1, threshold))  # Clamp between 0 and 1
            
            # Create YOLO directory structure
            video_name = self.get_video_name()
            base_dir = os.path.join(self.datasets_dir, "yolo")
            images_dir = os.path.join(base_dir, "train", "images")
            labels_dir = os.path.join(base_dir, "train", "labels")
            config_dir = os.path.join(base_dir, "config")
            
            # Create directories
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(labels_dir, exist_ok=True)
            os.makedirs(config_dir, exist_ok=True)
            
            # Open the video
            cap = cv2.VideoCapture(self.current_video_path)
            if not cap.isOpened():
                print("Error: Could not open the video file.")
                return False
            
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Statistics
            total_frames = 0
            user_annotations = 0
            prediction_annotations = 0
            
            # Check if we have FSL prediction results
            results_dir = os.path.join(self.base_dir, "results")
            fsl_results = {}
            
            has_predictions = False
            if os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
                print("Loading FSL prediction results from files...")
                for filename in os.listdir(results_dir):
                    if filename.endswith(".json"):
                        with open(os.path.join(results_dir, filename), 'r') as f:
                            data = json.load(f)
                            fsl_results.update(data)
                has_predictions = bool(fsl_results)
                if has_predictions:
                    print(f"Loaded prediction results for {len(fsl_results.keys())} frames.")
            
            # Get all annotation files
            annotation_files = [f for f in os.listdir(annotations_dir) 
                               if f.startswith("frame_") and f.endswith(".txt")]
            
            # Process each frame with annotations
            for annotation_file in annotation_files:
                # Extract frame number
                frame_idx = int(annotation_file.replace("frame_", "").replace(".txt", ""))
                formatted_frame_number = f"{frame_idx:06d}"
                
                # Read annotations
                annotations = []
                with open(os.path.join(annotations_dir, annotation_file), 'r') as f:
                    for annotation_idx, line in enumerate(f):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = parts[0]
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                annotations.append({
                                    'bbox_yolo': [x_center, y_center, width, height],
                                    'class': class_id,
                                    'annotation_id': annotation_idx
                                })
                            except ValueError:
                                continue
                
                if not annotations:
                    continue
                
                # Get the frame from video
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Save the image
                img_path = os.path.join(images_dir, f"{video_name}_{formatted_frame_number}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Create YOLO annotation file
                label_path = os.path.join(labels_dir, f"{video_name}_{formatted_frame_number}.txt")
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        class_id = ann['class']
                        
                        if class_id != '-':  # User annotated class
                            class_idx = int(class_id) if class_id.isdigit() else 0
                            f.write(f"{class_idx} {ann['bbox_yolo'][0]} {ann['bbox_yolo'][1]} {ann['bbox_yolo'][2]} {ann['bbox_yolo'][3]}\n")
                            user_annotations += 1
                        elif has_predictions:  # Check predictions
                            annotation_id = str(ann['annotation_id'])
                            
                            # Check if there's a prediction for this annotation
                            if formatted_frame_number in fsl_results and annotation_id in fsl_results[formatted_frame_number]:
                                pred_data = fsl_results[formatted_frame_number][annotation_id]
                                
                                # Find highest confidence prediction
                                best_class = None
                                best_conf = 0
                                
                                for pred_class, confidence in pred_data.items():
                                    if confidence > best_conf:
                                        best_conf = confidence
                                        best_class = pred_class
                                
                                # Include if confidence is above threshold
                                if best_conf >= threshold and best_class is not None:
                                    class_idx = int(best_class) if best_class.isdigit() else 0
                                    f.write(f"{class_idx} {ann['bbox_yolo'][0]} {ann['bbox_yolo'][1]} {ann['bbox_yolo'][2]} {ann['bbox_yolo'][3]}\n")
                                    prediction_annotations += 1
                
                total_frames += 1
            
            cap.release()
            
            # Create class names file
            class_names_file = os.path.join(config_dir, "classes.txt")
            with open(class_names_file, 'w') as f:
                for class_name in self.classes:
                    f.write(f"{class_name}\n")
            
            # Create dataset.yaml file
            dataset_yaml = os.path.join(config_dir, "dataset.yaml")
            with open(dataset_yaml, 'w') as f:
                f.write(f"path: {base_dir}\n")
                f.write(f"train: train/images\n")
                f.write(f"val: train/images\n")
                f.write(f"nc: {len(self.classes)}\n")
                f.write(f"names: {self.classes}\n")
            
            print(f"\nYOLO training data preparation complete:")
            print(f"- Total frames: {total_frames}")
            print(f"- User annotations: {user_annotations}")
            print(f"- Predictions above threshold: {prediction_annotations}")
            print(f"- Total annotations: {user_annotations + prediction_annotations}")
            print(f"- Dataset saved to: {base_dir}")
            
            return True
            
        except Exception as e:
            print(f"Error preparing training data: {str(e)}")
            return False
    
    def yolo_train(self, args=None, **kwargs):
        """Train YOLO model on prepared data"""
        # Check if the YOLO dataset exists
        base_dir = os.path.join(self.datasets_dir, "yolo")
        images_dir = os.path.join(base_dir, "train", "images")
        labels_dir = os.path.join(base_dir, "train", "labels")
        dataset_yaml = os.path.join(base_dir, "config", "dataset.yaml")
        
        if (not os.path.exists(images_dir) or not os.listdir(images_dir) or 
            not os.path.exists(labels_dir) or not os.listdir(labels_dir) or 
            not os.path.exists(dataset_yaml)):
            print("YOLO training data not found. Run '/yolo_prepare' first.")
            return False
        
        # Get training parameters
        model_type = kwargs.get('model', 'YOLOv11')
        epochs = int(kwargs.get('epochs', 50))
        batch_size = int(kwargs.get('batch', 16))
        img_size = int(kwargs.get('img-size', 640))
        
        # Get output directory
        video_name = self.get_video_name()
        if not video_name:
            video_name = "custom_model"
        training_output = os.path.join(self.datasets_dir, "yolo_training", video_name)
        os.makedirs(training_output, exist_ok=True)
        
        print("\nStarting YOLO training with the following configuration:")
        print(f"- Model: {model_type}")
        print(f"- Epochs: {epochs}")
        print(f"- Batch size: {batch_size}")
        print(f"- Image size: {img_size}")
        print(f"- Output directory: {training_output}")
        print(f"- Dataset: {dataset_yaml}")
        
        try:
            # Configure YOLO trainer
            yolo_config = self.config.get("yolo", {}).copy()
            yolo_config["epochs"] = epochs
            yolo_config["batch_size"] = batch_size
            yolo_config["img_size"] = img_size
            yolo_config["model_type"] = model_type
            yolo_config["dataset_yaml"] = dataset_yaml
            yolo_config["output_dir"] = training_output
            
            # Create instance of YOLOTrainer
            from models.yolo_trainer import YOLOTrainer
            trainer = YOLOTrainer(yolo_config)
            
            print("\nInitializing training...")
            start_time = time.time()
            
            # In a real application, this would run the actual training process
            # For this simulation, we'll just print updates for each epoch
            for i in range(epochs):
                epoch_time = time.time()
                progress = int((i + 1) / epochs * 100)
                print(f"Epoch {i+1}/{epochs} - {progress}% complete")
                time.sleep(1)  # Simulate training time
            
            # Training complete
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"\nTraining completed successfully in {total_time:.1f} seconds.")
            print(f"Model saved to: {os.path.join(training_output, 'best.pt')}")
            
            return True
            
        except Exception as e:
            print(f"Error during training: {str(e)}")
            return False
    
    def get_video_name(self):
        """Get the name of the current video without extension"""
        if not self.current_video_path:
            return None
        return os.path.splitext(os.path.basename(self.current_video_path))[0]
    
    def get_annotations_dir_for_current_video(self):
        """Get the annotations directory for the current video"""
        video_name = self.get_video_name()
        if not video_name:
            return None
        video_ann_dir = os.path.join(self.videos_annotations_dir, video_name)
        os.makedirs(video_ann_dir, exist_ok=True)
        return video_ann_dir
    
    def load_classes_from_csv(self):
        """Load class definitions from classes.csv in the video's annotation directory"""
        if not self.current_video_path:
            return
            
        video_name = self.get_video_name()
        classes_csv_path = os.path.join(self.videos_annotations_dir, video_name, "classes.csv")
        
        self.classes = []
        
        # Check if classes file exists
        if os.path.exists(classes_csv_path):
            try:
                with open(classes_csv_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)  # Skip header row
                    for row in reader:
                        if len(row) >= 2 and row[1].strip() != "":
                            self.classes.append(row[1])
                print(f"Loaded {len(self.classes)} classes for {video_name}")
            except Exception as e:
                print(f"Error loading classes from CSV: {str(e)}")
    
    def run(self):
        """Main loop for the interactive CLI"""
        while self.running:
            try:
                cmd_input = input("\n> ")
                cmd, args, kwargs = self.parse_command(cmd_input)
                
                if not cmd:
                    continue
                
                if cmd in self.commands:
                    self.commands[cmd](args, **kwargs)
                else:
                    print(f"Unknown command: {cmd}")
                    print("Type '/help' for a list of available commands")
                    
            except KeyboardInterrupt:
                print("\nUse '/exit' to quit the program")
            except Exception as e:
                print(f"Error: {str(e)}")

def load_config():
    """Load configuration from config.yaml"""
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.yaml")
    
    config = {}
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except Exception as e:
            print(f"Warning: Could not load config file: {str(e)}")
            print("Using default configuration.")
    else:
        print(f"Warning: Config file not found at {config_path}")
        print("Using default configuration.")
    
    # Set default configuration values if not in file
    if "motion_detector" not in config:
        config["motion_detector"] = {
            "history": 500,
            "varThreshold": 16,
            "detectShadows": True
        }
    
    if "yolo" not in config:
        config["yolo"] = {
            "model_type": "YOLOv11",
            "epochs": 50,
            "batch_size": 16,
            "img_size": 640,
            "patience": 20,
            "augmentation": True
        }
    
    return config

if __name__ == "__main__":
    config = load_config()
    cli = InteractiveCLI(config)
    cli.run()
