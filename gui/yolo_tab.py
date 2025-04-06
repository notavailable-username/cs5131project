from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QMessageBox, QGroupBox, QFileDialog, QProgressBar, QSpinBox,
    QCheckBox, QSplitter, QFormLayout, QTextEdit, QSlider
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import os
import cv2
import json
import shutil

class TrainingThread(QThread):
    """Thread for running YOLO training in the background"""
    progress_update = pyqtSignal(int)
    log_update = pyqtSignal(str)
    training_complete = pyqtSignal(bool, str)
    
    def __init__(self, trainer, training_dir, output_dir, config):
        super().__init__()
        self.trainer = trainer
        self.training_dir = training_dir
        self.output_dir = output_dir
        self.config = config
        
    def run(self):
        try:
            self.log_update.emit("Starting training...")
            # Here we would call the actual training method
            # For now, we just emit updates to simulate training
            self.log_update.emit(f"Training with {self.config['epochs']} epochs...")
            
            # In a real implementation, this would be replaced with actual training logic
            import time
            for i in range(self.config['epochs']):
                if self.isInterruptionRequested():
                    self.training_complete.emit(False, "Training was interrupted")
                    return
                    
                time.sleep(0.5)  # Simulate training time
                progress = int((i + 1) / self.config['epochs'] * 100)
                self.progress_update.emit(progress)
                self.log_update.emit(f"Completed epoch {i+1}/{self.config['epochs']}")
            
            # Training complete
            self.log_update.emit("Training completed successfully!")
            self.training_complete.emit(True, "Training completed successfully")
            
        except Exception as e:
            self.log_update.emit(f"Error during training: {str(e)}")
            self.training_complete.emit(False, f"Error: {str(e)}")

class YOLOTab(QWidget):
    """Tab for YOLO model training and management"""
    
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self.datasets_dir = parent.datasets_dir if hasattr(parent, 'datasets_dir') else os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        self.config = parent.config if hasattr(parent, 'config') else {}
        self.training_thread = None
        
        self._setup_ui()
        
    def _setup_ui(self):
        """Setup the UI components for the YOLO tab"""
        main_layout = QVBoxLayout()
        self.setLayout(main_layout)
        
        # Create a splitter for the tab
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # Left side - Training settings & controls
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # Settings group
        settings_group = QGroupBox("Training Settings")
        settings_layout = QFormLayout()
        
        # Model selection
        self.model_combo = QComboBox()
        self.model_combo.addItems(["YOLOv11", "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"])
        self.model_combo.setCurrentIndex(0)  # Set YOLOv11 as default
        self.model_combo.setToolTip("Select YOLO model variant (smaller is faster, larger is more accurate)")
        settings_layout.addRow("Model:", self.model_combo)
        
        # Epochs
        self.epochs_input = QSpinBox()
        self.epochs_input.setRange(1, 500)
        self.epochs_input.setValue(50)
        self.epochs_input.setToolTip("Number of epochs to train the model")
        settings_layout.addRow("Epochs:", self.epochs_input)
        
        # Batch size
        self.batch_size_input = QComboBox()
        self.batch_size_input.addItems(["1", "2", "4", "8", "16", "32", "64"])
        self.batch_size_input.setCurrentText("16")
        self.batch_size_input.setToolTip("Batch size for training (larger values require more GPU memory)")
        settings_layout.addRow("Batch Size:", self.batch_size_input)
        
        # Image size
        self.img_size_input = QComboBox()
        self.img_size_input.addItems(["320", "416", "512", "640", "1024"])
        self.img_size_input.setCurrentText("640")
        self.img_size_input.setToolTip("Input image size for training (larger values may be more accurate but slower)")
        settings_layout.addRow("Image Size:", self.img_size_input)
        
        # Patience for early stopping
        self.patience_input = QSpinBox()
        self.patience_input.setRange(5, 100)
        self.patience_input.setValue(20)
        self.patience_input.setToolTip("Number of epochs with no improvement after which training will be stopped")
        settings_layout.addRow("Patience:", self.patience_input)
        
        # Confidence threshold for including predictions
        self.threshold_input = QSpinBox()
        self.threshold_input.setRange(0, 100)
        self.threshold_input.setSuffix("%")
        self.threshold_input.setValue(50)  # Default 50%
        self.threshold_input.setToolTip("Only include predictions with confidence above this threshold")
        settings_layout.addRow("Confidence Threshold:", self.threshold_input)
        
        # Augmentation options
        self.augmentation_check = QCheckBox("Enable Data Augmentation")
        self.augmentation_check.setChecked(True)
        self.augmentation_check.setToolTip("Apply data augmentation to increase dataset variety")
        settings_layout.addRow(self.augmentation_check)
        
        # Add aspect ratio option
        self.preserve_aspect_ratio = QCheckBox("Preserve Aspect Ratio")
        self.preserve_aspect_ratio.setChecked(True)
        self.preserve_aspect_ratio.setToolTip("Maintain image aspect ratio when resizing (adds padding)")
        settings_layout.addRow(self.preserve_aspect_ratio)
        
        settings_group.setLayout(settings_layout)
        left_layout.addWidget(settings_group)
        
        # Data preparation group
        data_group = QGroupBox("Data Management")
        data_layout = QVBoxLayout()
        
        self.btn_prepare_training = QPushButton("Prepare Training Data")
        self.btn_prepare_training.clicked.connect(self.prepare_training_data)
        self.btn_prepare_training.setToolTip("Extract annotated frames and prepare data for YOLO training")
        data_layout.addWidget(self.btn_prepare_training)
        
        data_group.setLayout(data_layout)
        left_layout.addWidget(data_group)
        
        # Training controls group
        training_group = QGroupBox("Training Controls")
        training_layout = QVBoxLayout()
        
        self.btn_start_training = QPushButton("Start YOLO Training")
        self.btn_start_training.clicked.connect(self.start_yolo_training)
        self.btn_start_training.setToolTip("Begin training the YOLO model with prepared data")
        training_layout.addWidget(self.btn_start_training)
        
        self.btn_stop_training = QPushButton("Abort Training")
        self.btn_stop_training.clicked.connect(self.abort_training)
        self.btn_stop_training.setEnabled(False)
        self.btn_stop_training.setToolTip("Stop the current training process")
        training_layout.addWidget(self.btn_stop_training)
        
        self.training_progress = QProgressBar()
        self.training_progress.setValue(0)
        training_layout.addWidget(self.training_progress)
        
        training_group.setLayout(training_layout)
        left_layout.addWidget(training_group)
        
        # Add a stretch at the end to keep everything aligned at the top
        left_layout.addStretch()
        
        # Right side - Training log & results
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # Log viewer
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        
        self.log_output = QTextEdit()
        self.log_output.setReadOnly(True)
        log_layout.addWidget(self.log_output)
        
        log_group.setLayout(log_layout)
        right_layout.addWidget(log_group)
        
        # Results / Model Management
        results_group = QGroupBox("Model Management")
        results_layout = QVBoxLayout()
        
        self.btn_export_model = QPushButton("Export Trained Model")
        self.btn_export_model.clicked.connect(self.export_model)
        self.btn_export_model.setEnabled(False)
        results_layout.addWidget(self.btn_export_model)
        
        self.btn_inference = QPushButton("Test Model (Inference)")
        self.btn_inference.clicked.connect(self.test_inference)
        self.btn_inference.setEnabled(False)
        results_layout.addWidget(self.btn_inference)
        
        results_group.setLayout(results_layout)
        right_layout.addWidget(results_group)
        
        # Add a stretch at the end to keep everything aligned at the top
        right_layout.addStretch()
        
        # Add left and right panels to the splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        
        # Set initial splitter sizes (50/50)
        splitter.setSizes([500, 500])
    
    def prepare_training_data(self):
        """
        Prepares and organizes data for model training.
        Collects annotations and images from the current project,
        including both user annotations and few-shot predictions.
        """
        if not hasattr(self.parent, 'current_video_path') or not self.parent.current_video_path:
            QMessageBox.warning(self, "Warning", "Please select a video first.")
            return False
            
        if not hasattr(self.parent, 'classes') or len(self.parent.classes) == 0:
            QMessageBox.warning(self, "Warning", "Please define at least one class before preparing training data.")
            return False
        
        # Clear the log before starting
        self.log_output.clear()
        self.log_output.append("Starting to prepare training data...")
            
        # Create proper YOLO directory structure
        video_name = self.get_video_name()
        base_dir = os.path.join(self.datasets_dir, "yolo")
        images_dir = os.path.join(base_dir, "train", "images")
        labels_dir = os.path.join(base_dir, "train", "labels")
        config_dir = os.path.join(base_dir, "config")
        
        # Create directories
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        os.makedirs(config_dir, exist_ok=True)
        
        try:
            # Get confidence threshold (convert from percentage to decimal)
            threshold = self.threshold_input.value() / 100.0
            
            # Get annotations directory for the current video
            annotations_dir = self.parent.get_annotations_dir_for_current_video()
            if not annotations_dir or not os.path.exists(annotations_dir):
                QMessageBox.warning(self, "No Annotations", "No annotation directory found for this video.")
                return False
            
            # Open the video
            cap = cv2.VideoCapture(self.parent.current_video_path)
            if not cap.isOpened():
                QMessageBox.warning(self, "Error", "Could not open the video file.")
                return False
                
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Track statistics
            total_frames = 0
            total_annotations = 0
            user_annotations = 0
            prediction_annotations = 0
            
            # Check if we have FSL prediction results
            has_predictions = False
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            fsl_results = {}
            
            if hasattr(self.parent, 'fsl_tab') and hasattr(self.parent.fsl_tab, 'fsl_results'):
                has_predictions = True
                fsl_results = self.parent.fsl_tab.fsl_results
                self.log_output.append("Found FSL prediction results in memory.")
            elif os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
                # Load predictions from the results directory
                for filename in os.listdir(results_dir):
                    if filename.endswith(".json"):
                        with open(os.path.join(results_dir, filename), 'r') as f:
                            data = json.load(f)
                            fsl_results.update(data)
                has_predictions = bool(fsl_results)
                if has_predictions:
                    self.log_output.append("Loaded FSL prediction results from files.")
            
            # Get all annotation files
            annotation_files = [f for f in os.listdir(annotations_dir) 
                               if f.startswith("frame_") and f.endswith(".txt")]
            
            # Check class mapping
            class_map = {}
            if hasattr(self.parent, 'classes'):
                class_map = {str(i): class_name for i, class_name in enumerate(self.parent.classes)}
                class_map['-'] = '-'  # Add mapping for placeholder
            
            # Process each frame with annotations
            for annotation_file in annotation_files:
                # Extract frame number from filename
                frame_idx = int(annotation_file.replace("frame_", "").replace(".txt", ""))
                formatted_frame_number = f"{frame_idx:06d}"
                
                # Read in the annotations
                annotations = []
                with open(os.path.join(annotations_dir, annotation_file), 'r') as f:
                    annotation_idx = 0
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            try:
                                class_id = parts[0]
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                # Convert YOLO coordinates to absolute pixel coordinates for display
                                x1 = int((x_center - width / 2) * frame_width)
                                y1 = int((y_center - height / 2) * frame_height)
                                x2 = int((x_center + width / 2) * frame_width)
                                y2 = int((y_center + height / 2) * frame_height)
                                
                                # Create annotation object
                                annotation = {
                                    'bbox_abs': [max(0, x1), max(0, y1), min(frame_width - 1, x2), min(frame_height - 1, y2)],
                                    'class': class_id,
                                    'bbox_yolo': [x_center, y_center, width, height],
                                    'annotation_id': annotation_idx
                                }
                                
                                annotations.append(annotation)
                                annotation_idx += 1
                            except ValueError:
                                continue
                
                if not annotations:
                    continue
                    
                # Get the frame from video
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Save the image to YOLO images directory
                img_path = os.path.join(images_dir, f"{video_name}_{formatted_frame_number}.jpg")
                cv2.imwrite(img_path, frame)
                
                # Create YOLO format annotation file in labels directory
                label_path = os.path.join(labels_dir, f"{video_name}_{formatted_frame_number}.txt")
                with open(label_path, 'w') as f:
                    for ann in annotations:
                        class_id = ann['class']
                        
                        # Handle different types of annotations
                        if class_id != '-':  # User annotated class
                            # Use the class index directly
                            class_idx = int(class_id) if class_id.isdigit() else 0
                            f.write(f"{class_idx} {ann['bbox_yolo'][0]} {ann['bbox_yolo'][1]} {ann['bbox_yolo'][2]} {ann['bbox_yolo'][3]}\n")
                            user_annotations += 1
                        elif has_predictions:  # Check if we have predictions for this one
                            # This is a detection without user annotation, check FSL predictions
                            annotation_id = str(ann['annotation_id'])
                            
                            # Check if there's a prediction for this annotation
                            if formatted_frame_number in fsl_results and annotation_id in fsl_results[formatted_frame_number]:
                                pred_data = fsl_results[formatted_frame_number][annotation_id]
                                
                                # Find the highest confidence prediction
                                best_class = None
                                best_conf = 0
                                
                                for pred_class, confidence in pred_data.items():
                                    if confidence > best_conf:
                                        best_conf = confidence
                                        best_class = pred_class
                                
                                # Only include if confidence is above threshold
                                if best_conf >= threshold and best_class is not None:
                                    class_idx = int(best_class) if best_class.isdigit() else 0
                                    f.write(f"{class_idx} {ann['bbox_yolo'][0]} {ann['bbox_yolo'][1]} {ann['bbox_yolo'][2]} {ann['bbox_yolo'][3]}\n")
                                    prediction_annotations += 1
                
                total_frames += 1
                total_annotations += user_annotations + prediction_annotations
            
            cap.release()
            
            # Create class names file for YOLO in config directory
            class_names_file = os.path.join(config_dir, "classes.txt")
            with open(class_names_file, 'w') as f:
                for class_name in self.parent.classes:
                    f.write(f"{class_name}\n")
            
            # Create dataset.yaml file for YOLO in config directory
            dataset_yaml = os.path.join(config_dir, "dataset.yaml")
            with open(dataset_yaml, 'w') as f:
                f.write(f"path: {base_dir}\n")
                f.write(f"train: train/images\n")  # Updated path
                f.write(f"val: train/images\n")    # Updated path
                f.write(f"nc: {len(self.parent.classes)}\n")
                f.write(f"names: {self.parent.classes}\n")
            
            # Show statistics
            self.log_output.append(f"Training data preparation complete:")
            self.log_output.append(f"- Total frames: {total_frames}")
            self.log_output.append(f"- User annotations: {user_annotations}")
            self.log_output.append(f"- Predictions above threshold: {prediction_annotations}")
            self.log_output.append(f"- Total annotations: {user_annotations + prediction_annotations}")
            self.log_output.append(f"- Dataset saved to: {os.path.dirname(training_dir)}")
            
            # Show dialog with statistics
            QMessageBox.information(self, "Training Data Ready", 
                f"Training data prepared successfully!\n\n"
                f"Total frames: {total_frames}\n"
                f"User annotations: {user_annotations}\n"
                f"Predictions above {self.threshold_input.value()}% threshold: {prediction_annotations}\n"
                f"Total annotations: {user_annotations + prediction_annotations}")
            
            return True
            
        except Exception as e:
            self.log_output.append(f"Error preparing training data: {str(e)}")
            QMessageBox.critical(self, "Error", f"Failed to prepare training data: {str(e)}")
            return False
    
    def start_yolo_training(self):
        """
        Starts the YOLO model training process with the prepared data.
        """
        if not hasattr(self.parent, 'current_video_path') or not self.parent.current_video_path:
            QMessageBox.warning(self, "No Video Selected", "Please select a video first.")
            return
        
        video_name = self.get_video_name()
        images_dir = os.path.join(self.datasets_dir, "yolo", "train", "images")
        labels_dir = os.path.join(self.datasets_dir, "yolo", "train", "labels")
        
        # Check if training data exists
        if not os.path.exists(images_dir) or not os.listdir(images_dir) or not os.path.exists(labels_dir) or not os.listdir(labels_dir):
            QMessageBox.warning(self, "Missing Data", "Please prepare the training data first.")
            return
        
        # Get the path to dataset.yaml
        dataset_yaml = os.path.join(self.datasets_dir, "yolo", "config", "dataset.yaml")
        if not os.path.exists(dataset_yaml):
            QMessageBox.warning(self, "Missing Configuration", "Dataset configuration not found.")
            return
        
        # Configure YOLO trainer
        yolo_config = self.config.get("yolo", {}).copy()
        yolo_config["epochs"] = self.epochs_input.value()
        yolo_config["batch_size"] = int(self.batch_size_input.currentText())
        yolo_config["img_size"] = int(self.img_size_input.currentText())
        yolo_config["patience"] = self.patience_input.value()
        yolo_config["model_type"] = self.model_combo.currentText()
        yolo_config["augmentation"] = self.augmentation_check.isChecked()
        yolo_config["preserve_aspect_ratio"] = self.preserve_aspect_ratio.isChecked()
        yolo_config["dataset_yaml"] = dataset_yaml
        
        try:
            # Setup training output directory
            training_output = os.path.join(self.datasets_dir, "yolo_training", video_name)
            os.makedirs(training_output, exist_ok=True)
            
            # Pass the output directory to config
            yolo_config["output_dir"] = training_output
            
            # In a real implementation, you'd use the YOLOTrainer here
            from models.yolo_trainer import YOLOTrainer
            trainer = YOLOTrainer(yolo_config)
            
            # Clear log and update UI before training
            self.log_output.clear()
            self.log_output.append("Initializing training...")
            self.toggle_training_controls(True)
            
            # Start training in a thread
            self.training_thread = TrainingThread(trainer, training_dir, training_output, yolo_config)
            self.training_thread.progress_update.connect(self.training_progress.setValue)
            self.training_thread.log_update.connect(self.append_to_log)
            self.training_thread.training_complete.connect(self.on_training_complete)
            self.training_thread.start()
            
        except Exception as e:
            self.log_output.append(f"Error starting training: {str(e)}")
            QMessageBox.critical(self, "Training Error", f"Error starting training: {str(e)}")
            self.toggle_training_controls(False)
    
    def abort_training(self):
        """Stop the current training process"""
        if self.training_thread and self.training_thread.isRunning():
            self.log_output.append("Aborting training...")
            self.training_thread.requestInterruption()
            self.training_thread.wait(5000)  # Wait up to 5 seconds for clean shutdown
            
            if self.training_thread.isRunning():
                self.training_thread.terminate()
                self.log_output.append("Training terminated forcefully.")
            else:
                self.log_output.append("Training aborted successfully.")
            
            self.toggle_training_controls(False)
    
    def on_training_complete(self, success, message):
        """Handle the completion of training"""
        if success:
            self.log_output.append("Training completed successfully!")
            QMessageBox.information(self, "Training Complete", "YOLO training completed successfully.")
            
            # Enable model export and testing
            self.btn_export_model.setEnabled(True)
            self.btn_inference.setEnabled(True)
        else:
            self.log_output.append(f"Training failed: {message}")
            QMessageBox.warning(self, "Training Failed", f"Training process failed: {message}")
        
        # Re-enable UI controls
        self.toggle_training_controls(False)
    
    def toggle_training_controls(self, is_training):
        """Enable/disable UI controls during training"""
        # Training settings
        self.model_combo.setEnabled(not is_training)
        self.epochs_input.setEnabled(not is_training)
        self.batch_size_input.setEnabled(not is_training)
        self.img_size_input.setEnabled(not is_training)
        self.patience_input.setEnabled(not is_training)
        self.augmentation_check.setEnabled(not is_training)
        self.preserve_aspect_ratio.setEnabled(not is_training)
        
        # Control buttons
        self.btn_prepare_training.setEnabled(not is_training)
        self.btn_start_training.setEnabled(not is_training)
        self.btn_stop_training.setEnabled(is_training)
    
    def export_model(self):
        """Export the trained model to a selected directory"""
        if not hasattr(self.parent, 'current_video_path') or not self.parent.current_video_path:
            QMessageBox.warning(self, "No Video Selected", "Please select a video first.")
            return
            
        video_name = self.get_video_name()
        model_path = os.path.join(self.datasets_dir, "yolo_training", video_name, "best.pt")
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Model Not Found", "Trained model not found. Train a model first.")
            return
            
        # Get export directory from user
        export_dir = QFileDialog.getExistingDirectory(self, "Select Directory to Export Model")
        if not export_dir:
            return
            
        try:
            # Copy the model file
            export_path = os.path.join(export_dir, f"yolo_{video_name}.pt")
            shutil.copy2(model_path, export_path)
            
            # Also export classes file
            classes_path = os.path.join(self.datasets_dir, "training_data", video_name, "classes.txt")
            if os.path.exists(classes_path):
                shutil.copy2(classes_path, os.path.join(export_dir, f"classes_{video_name}.txt"))
                
            self.log_output.append(f"Model exported to: {export_path}")
            QMessageBox.information(self, "Export Successful", f"Model exported to:\n{export_path}")
            
        except Exception as e:
            self.log_output.append(f"Error exporting model: {str(e)}")
            QMessageBox.critical(self, "Export Error", f"Failed to export model: {str(e)}")
    
    def test_inference(self):
        """Test the trained model on images/video"""
        if not hasattr(self.parent, 'current_video_path') or not self.parent.current_video_path:
            QMessageBox.warning(self, "No Video Selected", "Please select a video first.")
            return
            
        video_name = self.get_video_name()
        model_path = os.path.join(self.datasets_dir, "yolo_training", video_name, "best.pt")
        
        if not os.path.exists(model_path):
            QMessageBox.warning(self, "Model Not Found", "Trained model not found. Train a model first.")
            return
        
        # For demonstration purposes, we'll show a mock inference result
        # In a real application, this would use the YOLO model to run inference
        self.log_output.append("Running inference on current video...")
        
        # Function would use model to detect objects
        # For now we just show a message
        QMessageBox.information(self, "Inference Test", 
                             "Model inference would be performed here.\n"
                             "In a full implementation, this would run the model on the current video or selected images.")
    
    def append_to_log(self, message):
        """Add a message to the log output"""
        self.log_output.append(message)
        
    def get_video_name(self):
        """Get the name of the current video without extension"""
        if not hasattr(self.parent, 'current_video_path') or not self.parent.current_video_path:
            return None
        return os.path.splitext(os.path.basename(self.parent.current_video_path))[0]
