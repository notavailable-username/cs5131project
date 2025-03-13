from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QPushButton, 
    QFileDialog, QListWidget, QLabel, QMessageBox, QProgressBar, QComboBox,
    QDockWidget, QStackedLayout, QSplitter, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import cv2
import os
import numpy as np
import torch
from .video_player import VideoPlayer
from .image_annotator_dialog import ImageAnnotatorDialog
from models.motion_detector import MotionDetector
from models.few_shot import FewShotEnsemble
from models.yolo_trainer import YOLOTrainer
import json
from PyQt6.QtCore import QPoint, QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
                             QLabel, QPushButton, QComboBox, QFileDialog,
                             QInputDialog, QMessageBox, QScrollArea)

class VideoProcessThread(QThread):
    update_progress = pyqtSignal(int)
    update_frame = pyqtSignal(object)
    detection_complete = pyqtSignal(list, list)
    
    def __init__(self, video_path, motion_detector, few_shot, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.motion_detector = motion_detector
        self.few_shot = few_shot
        self.running = True
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = max(1, int(fps / 2))  # Process 2 frames per second
        
        all_detections = []
        uncertain_frames = []
        frame_count = 0
        
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every nth frame
            if frame_count % frame_interval == 0:
                # Update progress
                progress = int(100 * frame_count / total_frames)
                self.update_progress.emit(progress)
                
                # Send frame to GUI for display
                self.update_frame.emit(frame.copy())
                
                # Run motion detection
                boxes = self.motion_detector.detect(frame)
                
                if boxes:
                    frame_detections = []
                    frame_is_uncertain = False
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        # Ensure coordinates are within frame
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                        
                        if x2 > x1 and y2 > y1:
                            # Extract patch and run few-shot prediction
                            patch = frame[y1:y2, x1:x2]
                            label, conf = self.few_shot.predict(patch)
                            
                            # Record detection 
                            detection = {
                                'frame_idx': frame_count,
                                'bbox': box,
                                'class': label,
                                'confidence': conf,
                                'timestamp': frame_count / fps
                            }
                            frame_detections.append(detection)
                            
                            # Flag low confidence detections
                            if conf < 0.7 or label == "uncertain":
                                frame_is_uncertain = True
                    
                    # Save results
                    if frame_detections:
                        all_detections.extend(frame_detections)
                        if frame_is_uncertain:
                            uncertain_frames.append((frame_count, frame.copy()))
            
            frame_count += 1
            
            # Check if thread should stop
            if not self.running:
                break
                
        cap.release()
        self.detection_complete.emit(all_detections, uncertain_frames)
    
    def stop(self):
        self.running = False


class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Motion-Aware Few-Shot Object Detection")
        self.resize(1200, 800)
        
        # Initialize our models
        md_conf = config.get("motion_detector", {})
        self.motion_detector = MotionDetector(**md_conf)
        
        fs_conf = config.get("few_shot", {})
        self.few_shot = FewShotEnsemble(confidence_threshold=fs_conf.get("confidence_threshold", 0.5))
        
        # Data storage
        self.video_path = None
        self.output_dir = None
        self.all_detections = []
        self.uncertain_frames = []
        self.support_examples = {}  # {class_name: [image_patches]}
        self.classes = []
        
        self.process_thread = None
        
        self._init_ui()
    
    def _init_ui(self):
        # Create menu bar
        self._create_menu_bar()
        
        # Main splitter to divide the screen
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.setCentralWidget(self.main_splitter)
        
        # Left panel: Video/Image display
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_widget.setLayout(left_layout)
        
        # Video player and controls
        self.video_player = VideoPlayer()
        left_layout.addWidget(self.video_player)
        
        # Add left panel to splitter
        self.main_splitter.addWidget(left_widget)
        
        # Right panel: Tabs for different functionalities
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        self.tabs = QTabWidget()
        # Note: Import tab removed
        self.tabs.addTab(self._create_detection_tab(), "Detection")
        self.tabs.addTab(self._create_few_shot_tab(), "Few-Shot Learning")
        self.tabs.addTab(self._create_annotation_tab(), "Manual Annotation")
        self.tabs.addTab(self._create_training_tab(), "YOLO Training")
        
        right_layout.addWidget(self.tabs)
        
        # Add right panel to splitter
        self.main_splitter.addWidget(right_widget)
        
        # Set the initial size ratio
        self.main_splitter.setSizes([600, 600])
        
        # Timer for updating video
        self.update_timer = QTimer()
        self.update_timer.timeout.connect(self.update_video_display)
    
    def _create_menu_bar(self):
        """Create the application menu bar"""
        menubar = self.menuBar()
        
        # File menu
        file_menu = menubar.addMenu("File")
        
        # Import video actions
        import_video_action = file_menu.addAction("Import Video")
        import_video_action.triggered.connect(self.import_video)
        
        import_videos_dir_action = file_menu.addAction("Import All Videos in Directory")
        import_videos_dir_action.triggered.connect(self.import_videos_directory)
        
        file_menu.addSeparator()
        
        # Import image actions
        import_image_action = file_menu.addAction("Import Image")
        import_image_action.triggered.connect(self.import_image)
        
        import_images_dir_action = file_menu.addAction("Import All Images in Directory")
        import_images_dir_action.triggered.connect(self.import_images_directory)
        
        file_menu.addSeparator()
        
        # Output directory selection
        select_output_action = file_menu.addAction("Select Output Directory")
        select_output_action.triggered.connect(self.select_output_dir)
        
        # Exit action
        file_menu.addSeparator()
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # Edit menu (for future expansion)
        edit_menu = menubar.addMenu("Edit")
        
        # Class management action
        manage_classes_action = edit_menu.addAction("Manage Classes")
        manage_classes_action.triggered.connect(self.manage_classes)
    
    # Remove the _create_import_tab method as it's no longer needed
    
    def import_video(self):
        """Import a single video file to datasets/videos"""
        video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if not video_path:
            return
            
        # Create the datasets/videos directory if it doesn't exist
        videos_dir = os.path.join("datasets", "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        # Copy the video to the datasets/videos directory
        filename = os.path.basename(video_path)
        destination = os.path.join(videos_dir, filename)
        
        try:
            import shutil
            shutil.copy2(video_path, destination)
            self.video_path = destination
            self.video_player.load_video(destination)
            QMessageBox.information(self, "Video Imported", f"Video imported: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import video: {str(e)}")
    
    def import_videos_directory(self):
        """Import all videos from a directory to datasets/videos"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Videos")
        
        if not directory:
            return
            
        # Create the datasets/videos directory if it doesn't exist
        videos_dir = os.path.join("datasets", "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        # Get all video files from the directory
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        imported_count = 0
        
        try:
            import shutil
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
                    destination = os.path.join(videos_dir, file)
                    shutil.copy2(file_path, destination)
                    imported_count += 1
            
            if imported_count > 0:
                QMessageBox.information(self, "Videos Imported", f"Imported {imported_count} videos to datasets/videos")
                # Load the first video if we haven't loaded any yet
                if not self.video_path and imported_count > 0:
                    first_video = next(iter([os.path.join(videos_dir, f) for f in os.listdir(videos_dir) 
                                         if any(f.lower().endswith(ext) for ext in video_extensions)]), None)
                    if first_video:
                        self.video_path = first_video
                        self.video_player.load_video(first_video)
            else:
                QMessageBox.information(self, "No Videos Found", "No video files found in the selected directory.")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import videos: {str(e)}")
    
    def import_image(self):
        """Import a single image file to datasets/images"""
        image_path, _ = QFileDialog.getOpenFileName(
            self, "Select Image", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        
        if not image_path:
            return
            
        # Create the datasets/images directory if it doesn't exist
        images_dir = os.path.join("datasets", "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Copy the image to the datasets/images directory
        filename = os.path.basename(image_path)
        destination = os.path.join(images_dir, filename)
        
        try:
            import shutil
            shutil.copy2(image_path, destination)
            QMessageBox.information(self, "Image Imported", f"Image imported: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import image: {str(e)}")
    
    def import_images_directory(self):
        """Import all images from a directory to datasets/images"""
        directory = QFileDialog.getExistingDirectory(self, "Select Directory with Images")
        
        if not directory:
            return
            
        # Create the datasets/images directory if it doesn't exist
        images_dir = os.path.join("datasets", "images")
        os.makedirs(images_dir, exist_ok=True)
        
        # Get all image files from the directory
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
        imported_count = 0
        
        try:
            import shutil
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
                    destination = os.path.join(images_dir, file)
                    shutil.copy2(file_path, destination)
                    imported_count += 1
            
            if imported_count > 0:
                QMessageBox.information(self, "Images Imported", f"Imported {imported_count} images to datasets/images")
            else:
                QMessageBox.information(self, "No Images Found", "No image files found in the selected directory.")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import images: {str(e)}")
    
    def _create_detection_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Motion detection
        btn_run_detection = QPushButton("Run Motion Detection")
        btn_run_detection.clicked.connect(self.run_motion_detection)
        layout.addWidget(btn_run_detection)
        
        # Detection settings
        self.detection_threshold = QComboBox()
        self.detection_threshold.addItems(["Low", "Medium", "High"])
        self.detection_threshold.setCurrentIndex(1)
        layout.addWidget(QLabel("Detection Sensitivity:"))
        layout.addWidget(self.detection_threshold)
        
        # Progress indicator
        self.detection_progress = QProgressBar()
        layout.addWidget(self.detection_progress)
        
        # Results summary
        self.detection_summary = QLabel("No detections yet")
        layout.addWidget(self.detection_summary)
        
        widget.setLayout(layout)
        return widget
    
    def _create_few_shot_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Support set selection
        layout.addWidget(QLabel("Select support examples for few-shot learning:"))
        
        self.class_selector = QComboBox()
        layout.addWidget(self.class_selector)
        
        btn_add_support = QPushButton("Add Current Frame as Support Example")
        btn_add_support.clicked.connect(self.add_support_example)
        layout.addWidget(btn_add_support)
        
        # Support examples list
        layout.addWidget(QLabel("Current support examples:"))
        self.support_list = QListWidget()
        layout.addWidget(self.support_list)
        
        # Few-shot classification
        btn_run_few_shot = QPushButton("Run Few-Shot Classification")
        btn_run_few_shot.clicked.connect(self.run_few_shot_classification)
        layout.addWidget(btn_run_few_shot)
        
        widget.setLayout(layout)
        return widget
    
    def _create_annotation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Open label tool button
        btn_open_labeler = QPushButton("Open Label Tool")
        btn_open_labeler.clicked.connect(self.open_label_tool)
        layout.addWidget(btn_open_labeler)
        
        # Uncertain frames
        layout.addWidget(QLabel("Frames with low confidence predictions:"))
        self.uncertain_frames_list = QListWidget()
        self.uncertain_frames_list.itemClicked.connect(self.show_uncertain_frame)
        layout.addWidget(self.uncertain_frames_list)
        
        widget.setLayout(layout)
        return widget
    
    def _create_training_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Training settings
        layout.addWidget(QLabel("YOLO Training Settings:"))
        
        self.epochs_input = QComboBox()
        self.epochs_input.addItems(["10", "20", "50", "100"])
        layout.addWidget(QLabel("Training Epochs:"))
        layout.addWidget(self.epochs_input)
        
        self.batch_size_input = QComboBox()
        self.batch_size_input.addItems(["4", "8", "16", "32"])
        layout.addWidget(QLabel("Batch Size:"))
        layout.addWidget(self.batch_size_input)
        
        # Training controls
        btn_prepare_training = QPushButton("Prepare Training Data")
        btn_prepare_training.clicked.connect(self.prepare_training_data)
        layout.addWidget(btn_prepare_training)
        
        btn_start_training = QPushButton("Start YOLO Training")
        btn_start_training.clicked.connect(self.start_yolo_training)
        layout.addWidget(btn_start_training)
        
        self.training_status = QLabel("Training not started")
        layout.addWidget(self.training_status)
        
        widget.setLayout(layout)
        return widget
    
    def load_video(self):
        self.video_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        if self.video_path:
            self.video_player.load_video(self.video_path)
            QMessageBox.information(self, "Video Loaded", f"Video loaded: {os.path.basename(self.video_path)}")
    
    def load_images(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            # TODO: Implementation for handling image directories
            self.output_dir = directory
            self.output_dir_label.setText(f"Output directory: {directory}")
            QMessageBox.information(self, "Directory Selected", f"Image directory: {directory}")
    
    def select_output_dir(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_dir_label.setText(f"Output directory: {directory}")
    
    def manage_classes(self):
        # Open the class management dialog (similar to LabelingTool class functionality)
        # For now, we'll just use a simple list
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        self.class_selector.clear()
        self.class_selector.addItems(self.classes)
        QMessageBox.information(self, "Classes Loaded", "Default classes loaded")
    
    def toggle_video_playback(self):
        if self.video_player.is_playing():
            self.video_player.pause()
        else:
            self.video_player.play()
    
    def update_video_display(self):
        # Update the video display with overlay annotations
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.video_player.set_frame(self.current_frame)
    
    def run_motion_detection(self):
        if not self.video_path or not self.output_dir:
            QMessageBox.warning(self, "Missing Input", "Please select both a video and output directory.")
            return
        
        # Update motion detector settings based on UI
        sensitivity = self.detection_threshold.currentText()
        if sensitivity == "Low":
            self.motion_detector.varThreshold = 25
        elif sensitivity == "Medium":
            self.motion_detector.varThreshold = 16
        elif sensitivity == "High":
            self.motion_detector.varThreshold = 10
        
        # Start processing thread
        self.process_thread = VideoProcessThread(
            self.video_path, self.motion_detector, self.few_shot
        )
        self.process_thread.update_progress.connect(self.detection_progress.setValue)
        self.process_thread.update_frame.connect(self.update_detection_display)
        self.process_thread.detection_complete.connect(self.handle_detection_complete)
        
        self.detection_progress.setValue(0)
        self.process_thread.start()
    
    def update_detection_display(self, frame):
        self.current_frame = frame
        # Draw bounding boxes from motion detector
        boxes = self.motion_detector.detect(frame)
        for box in boxes:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Convert to Qt format and display
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.video_player.set_image(frame)
    
    def handle_detection_complete(self, all_detections, uncertain_frames):
        self.all_detections = all_detections
        self.uncertain_frames = uncertain_frames
        
        # Update the UI
        self.detection_summary.setText(f"Detected {len(all_detections)} objects in {len(set([d['frame_idx'] for d in all_detections]))} frames")
        
        # Update the uncertain frames list
        self.uncertain_frames_list.clear()
        for frame_idx, _ in uncertain_frames:
            self.uncertain_frames_list.addItem(f"Frame {frame_idx}")
        
        # Save detection results
        self.save_detection_results()
        
        QMessageBox.information(self, "Detection Complete", 
                              f"Motion detection completed.\n"
                              f"Total detections: {len(all_detections)}\n"
                              f"Uncertain frames: {len(uncertain_frames)}")
    
    def save_detection_results(self):
        if not self.output_dir:
            return
            
        # Create detection directory
        detection_dir = os.path.join(self.output_dir, "detections")
        os.makedirs(detection_dir, exist_ok=True)
        
        # Save all detection frames
        for i, (frame_idx, frame) in enumerate(self.uncertain_frames):
            frame_path = os.path.join(detection_dir, f"uncertain_frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
    
    def show_uncertain_frame(self, item):
        index = self.uncertain_frames_list.row(item)
        if index < len(self.uncertain_frames):
            _, frame = self.uncertain_frames[index]
            self.video_player.set_image(frame)
    
    def add_support_example(self):
        # Get current class
        current_class = self.class_selector.currentText()
        if not current_class:
            QMessageBox.warning(self, "No Class Selected", "Please select a class first.")
            return
            
        # Get current frame
        current_frame = self.video_player.get_current_frame()
        if current_frame is None:
            QMessageBox.warning(self, "No Frame", "No frame is currently displayed.")
            return
            
        # Add to support examples
        if current_class not in self.support_examples:
            self.support_examples[current_class] = []
        
        self.support_examples[current_class].append(current_frame)
        
        # Update UI
        self.support_list.addItem(f"{current_class} - Example {len(self.support_examples[current_class])}")
        
        QMessageBox.information(self, "Example Added", 
                              f"Added support example for class '{current_class}'.\n"
                              f"Total examples for this class: {len(self.support_examples[current_class])}")
    
    def run_few_shot_classification(self):
        if not self.support_examples:
            QMessageBox.warning(self, "No Support Examples", "Please add support examples for few-shot learning.")
            return
            
        # Train few-shot models with support examples
        for class_name, examples in self.support_examples.items():
            self.few_shot.add_support_examples(class_name, examples)
            
        self.few_shot.train()
        
        # Re-run classification on uncertain frames
        for i, (frame_idx, frame) in enumerate(self.uncertain_frames):
            # Get boxes from motion detector
            boxes = self.motion_detector.detect(frame)
            
            # Display frame with updated classifications
            display_frame = frame.copy()
            for box in boxes:
                x1, y1, x2, y2 = box
                patch = frame[y1:y2, x1:x2]
                label, conf = self.few_shot.predict(patch)
                
                # Draw box and label
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"{label} ({conf:.2f})", 
                          (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save classified frame
            classified_dir = os.path.join(self.output_dir, "classified")
            os.makedirs(classified_dir, exist_ok=True)
            cv2.imwrite(os.path.join(classified_dir, f"classified_{frame_idx}.jpg"), display_frame)
        
        QMessageBox.information(self, "Classification Complete", 
                              "Few-shot classification completed.\nResults saved to output directory.")
    
    def open_label_tool(self):
        if not self.output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory first.")
            return
            
        # Create directory for uncertain frames if it doesn't exist
        uncertain_dir = os.path.join(self.output_dir, "uncertain")
        os.makedirs(uncertain_dir, exist_ok=True)
        
        # Save uncertain frames for labeling
        for i, (frame_idx, frame) in enumerate(self.uncertain_frames):
            frame_path = os.path.join(uncertain_dir, f"frame_{frame_idx}.jpg")
            cv2.imwrite(frame_path, frame)
        
        # Using the separated ImageAnnotatorDialog instead of LabelDialog
        self.open_integrated_label_tool(uncertain_dir)

    def set_label_directory(self, directory):
        # This method now directly handles the directory without using an external tool
        if hasattr(self, 'annotator_dialog') and self.annotator_dialog.isVisible():
            self.annotator_dialog.load_directory(directory)
        else:
            self.open_integrated_label_tool(directory)
    
    def open_integrated_label_tool(self, directory=None):
        """Open an integrated labeling tool dialog"""
        self.annotator_dialog = ImageAnnotatorDialog(self, directory)
        self.annotator_dialog.finished.connect(self.on_labeling_finished)
        self.annotator_dialog.show()
    
    def on_labeling_finished(self):
        # Handle any post-labeling tasks here
        QMessageBox.information(self, "Labeling Complete", 
                               "Image labeling completed. Annotations saved.")
        
        # You might want to reload or update something here
        # For example, refresh the model with new training data
        pass

    def prepare_training_data(self):
        """
        Prepares and organizes data for model training.
        Collects annotations and images from the current project.
        """
        if not self.output_dir:
            QMessageBox.warning(self, "Warning", "Please select an output directory first.")
            return False
            
        if len(self.classes) == 0:
            QMessageBox.warning(self, "Warning", "Please define at least one class before preparing training data.")
            return False
            
        # Create training data directory structure
        training_dir = os.path.join(self.output_dir, "training_data")
        os.makedirs(training_dir, exist_ok=True)
        
        # Process annotations and organize files
        try:
            # Check if we have detection results to use
            if not self.all_detections:
                QMessageBox.warning(self, "Warning", "No detection results available. Run detection first.")
                return False
                
            # Copy annotated images and their labels to the training directory
            images_count = 0
            for i, detections in enumerate(self.all_detections):
                if detections:  # If the frame has detections
                    # Get the frame image
                    cap = cv2.VideoCapture(self.video_path)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        # Save image
                        img_path = os.path.join(training_dir, f"image_{i:06d}.jpg")
                        cv2.imwrite(img_path, frame)
                        
                        # Save annotations in YOLO format
                        label_path = os.path.join(training_dir, f"image_{i:06d}.txt")
                        with open(label_path, 'w') as f:
                            for det in detections:
                                class_idx = self.classes.index(det['class']) if 'class' in det else 0
                                x, y, w, h = det['bbox']
                                # Convert to YOLO format (normalized)
                                height, width = frame.shape[:2]
                                x_center = (x + w/2) / width
                                y_center = (y + h/2) / height
                                w_norm = w / width
                                h_norm = h / height
                                f.write(f"{class_idx} {x_center} {y_center} {w_norm} {h_norm}\n")
                        
                        images_count += 1
            
            # Create class mapping file
            with open(os.path.join(training_dir, "classes.txt"), 'w') as f:
                for class_name in self.classes:
                    f.write(f"{class_name}\n")
                    
            QMessageBox.information(self, "Success", f"Training data prepared successfully with {images_count} images.")
            return True
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to prepare training data: {str(e)}")
            return False

    def start_yolo_training(self):
        """
        Starts the YOLO model training process with the prepared data.
        """
        if not self.output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory first.")
            return
            
        training_dir = os.path.join(self.output_dir, "training_data")
        if not os.path.exists(training_dir) or not os.listdir(training_dir):
            QMessageBox.warning(self, "Missing Data", "Please prepare the training data first.")
            return
        
        # Configure YOLO trainer
        yolo_config = self.config.get("yolo", {})
        yolo_config["epochs"] = int(self.epochs_input.currentText())
        yolo_config["batch_size"] = int(self.batch_size_input.currentText())
        
        try:
            trainer = YOLOTrainer(yolo_config)
            
            # Setup training output directory
            training_output = os.path.join(self.output_dir, "yolo_training")
            os.makedirs(training_output, exist_ok=True)
            
            # Start training in a separate thread (simplified for now)
            self.training_status.setText("Training started...")
            
            # In a real implementation, you'd run this in a QThread with progress updates
            # For now, we're just showing a message about the intended behavior
            QMessageBox.information(
                self, "Training Started", 
                f"YOLO training started with {yolo_config['epochs']} epochs.\n"
                f"Results will be saved to {training_output}"
            )
            
            # TODO: Implement actual training in a thread with progress updates
            # self.training_thread = YOLOTrainingThread(trainer, training_dir, training_output)
            # self.training_thread.progress_update.connect(self.update_training_progress)
            # self.training_thread.finished.connect(self.training_complete)
            # self.training_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"Error starting training: {str(e)}")
