from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QPushButton, 
    QFileDialog, QListWidget, QLabel, QMessageBox, QProgressBar, QComboBox,
    QDockWidget, QStackedLayout, QSplitter, QTableWidget, QTableWidgetItem
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import cv2
import os
import numpy as np
import torch
from .video_player import VideoPlayer
from models.motion_detector import MotionDetector
from models.few_shot import FewShotEnsemble
from models.yolo_trainer import YOLOTrainer
from .image_labeller import LabelingTool

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
        
        # Video controls and progress
        control_layout = QHBoxLayout()
        self.btn_play_pause = QPushButton("Play")
        self.btn_play_pause.clicked.connect(self.toggle_video_playback)
        control_layout.addWidget(self.btn_play_pause)
        
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        left_layout.addLayout(control_layout)
        
        # Add left panel to splitter
        self.main_splitter.addWidget(left_widget)
        
        # Right panel: Tabs for different functionalities
        right_widget = QWidget()
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_import_tab(), "Import")
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
        
    def _create_import_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Media loading buttons
        btn_load_video = QPushButton("Load Video")
        btn_load_video.clicked.connect(self.load_video)
        layout.addWidget(btn_load_video)
        
        btn_load_images = QPushButton("Load Images")
        btn_load_images.clicked.connect(self.load_images)
        layout.addWidget(btn_load_images)
        
        # Output directory
        btn_output_dir = QPushButton("Select Output Directory")
        btn_output_dir.clicked.connect(self.select_output_dir)
        layout.addWidget(btn_output_dir)
        
        self.output_dir_label = QLabel("Output directory: Not selected")
        layout.addWidget(self.output_dir_label)
        
        # Class management
        btn_manage_classes = QPushButton("Manage Classes")
        btn_manage_classes.clicked.connect(self.manage_classes)
        layout.addWidget(btn_manage_classes)
        
        widget.setLayout(layout)
        return widget
    
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
            self.btn_play_pause.setText("Play")
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
            self.btn_play_pause.setText("Play")
        else:
            self.video_player.play()
            self.btn_play_pause.setText("Pause")
    
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
        
        # Launch labeling tool
        self.label_tool = LabelingTool()
        self.label_tool.show()
        
        # Set labeling tool to use the uncertain frames directory
        QTimer.singleShot(500, lambda: self.set_label_directory(uncertain_dir))

    def set_label_directory(self, directory):
        # This method serves as a bridge to the labeling tool's method
        if hasattr(self, 'label_tool'):
            self.label_tool.load_image_directory()  # Trigger the file dialog
            # In a real implementation, you'd modify LabelingTool to accept a directory parameter
    
    def prepare_training_data(self):
        if not self.output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory first.")
            return
        
        # Create YOLO dataset structure
        dataset_dir = os.path.join(self.output_dir, "yolo_dataset")
        os.makedirs(dataset_dir, exist_ok=True)
        
        # Create train, val directories
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)
        
        # Create images and labels subdirectories
        for d in [train_dir, val_dir]:
            os.makedirs(os.path.join(d, "images"), exist_ok=True)
            os.makedirs(os.path.join(d, "labels"), exist_ok=True)
        
        # Copy images and annotations
        # TODO: Implement actual data preparation
        
        # Create dataset.yaml
        with open(os.path.join(dataset_dir, "dataset.yaml"), "w") as f:
            f.write(f"path: {dataset_dir}\n")
            f.write(f"train: train/images\n")
            f.write(f"val: val/images\n\n")
            f.write(f"nc: {len(self.classes)}\n")
            f.write(f"names: {self.classes}\n")
        
        self.training_status.setText("Training data prepared")
        QMessageBox.information(self, "Data Prepared", "YOLO training data has been prepared.")
    
    def start_yolo_training(self):
        if not self.output_dir:
            QMessageBox.warning(self, "No Output Directory", "Please select an output directory first.")
            return
            
        dataset_yaml = os.path.join(self.output_dir, "yolo_dataset", "dataset.yaml")
        if not os.path.exists(dataset_yaml):
            QMessageBox.warning(self, "Missing Dataset", "Please prepare the training data first.")
            return
        
        # Configure YOLO trainer
        yolo_config = self.config.get("yolo", {})
        yolo_config["epochs"] = int(self.epochs_input.currentText())
        yolo_config["batch_size"] = int(self.batch_size_input.currentText())
        
        trainer = YOLOTrainer(yolo_config)
        
        # Start training in a separate thread
        self.training_status.setText("Training started...")
        # TODO: Implement actual training in a thread
        
        QMessageBox.information(self, "Training Started", 
                              f"YOLO training started with {yolo_config['epochs']} epochs.\n"
                              f"Results will be saved to {self.output_dir}/yolo_training")

