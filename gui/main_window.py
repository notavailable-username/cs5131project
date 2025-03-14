from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QPushButton, 
    QFileDialog, QListWidget, QLabel, QMessageBox, QProgressBar, QComboBox,
    QDockWidget, QStackedLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QSpinBox, QDoubleSpinBox, QSlider, QSizePolicy
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
    update_frame_number = pyqtSignal(int)  # New signal for frame number updates
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
                # Make sure we signal 100% progress when finished
                self.update_progress.emit(100)
                break
                
            # Process every nth frame
            if frame_count % frame_interval == 0:
                # Update progress and frame number - calculate progress correctly
                progress = min(100, int(100 * frame_count / total_frames))
                self.update_progress.emit(progress)
                self.update_frame_number.emit(frame_count)
                
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
                            
                            # Record detection with actual confidence
                            detection = {
                                'frame_idx': frame_count,
                                'bbox': box,
                                'class': label,
                                'confidence': float(conf),  # Ensure confidence is a float
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
        self.all_detections = []
        self.uncertain_frames = []
        self.support_examples = {}  # {class_name: [image_patches]}
        self.classes = []
        
        # Create base directories
        self.datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        self.annotations_dir = os.path.join(self.datasets_dir, "annotations")
        os.makedirs(self.annotations_dir, exist_ok=True)
        
        # Create directory for uncertain frames
        self.uncertain_dir = os.path.join(self.datasets_dir, "uncertain", "motion_detector")
        os.makedirs(self.uncertain_dir, exist_ok=True)
        
        self.process_thread = None
        self.detection_running = False
        
        self._init_ui()
        self._apply_styles()
    
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
    
    def _apply_styles(self):
        """Apply consistent styling to buttons and controls throughout the application"""
        # Button style: white text/icons with slightly lighter background
        button_style = """
            QPushButton {
                color: white;
                background-color: #3d3d3d;
                border: none;
                padding: 5px;
                border-radius: 3px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
            QPushButton:disabled {
                background-color: #555555;
                color: #888888;
            }
        """
        
        # Spinbox and ComboBox style
        spinbox_style = """
            QSpinBox, QDoubleSpinBox, QComboBox {
                color: white;
                background-color: #3d3d3d;
                border: 1px solid #555555;
                border-radius: 2px;
            }
            QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {
                background-color: #4a4a4a;
            }
            QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {
                background-color: #555555;
                color: #888888;
            }
        """
        
        # Apply styles to all buttons in this window
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)
        
        # Apply styles to all spinboxes and combo boxes
        for spinbox in self.findChildren(QSpinBox):
            spinbox.setStyleSheet(spinbox_style)
        
        for dspinbox in self.findChildren(QDoubleSpinBox):
            dspinbox.setStyleSheet(spinbox_style)
            
        for combobox in self.findChildren(QComboBox):
            combobox.setStyleSheet(spinbox_style)
    
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
        
        # Exit action
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
        
        # Edit menu (for future expansion)
        edit_menu = menubar.addMenu("Edit")
        
        # Class management action
        manage_classes_action = edit_menu.addAction("Manage Classes")
        manage_classes_action.triggered.connect(self.manage_classes)
    
    def _create_detection_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Group 1: Motion detection settings
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        
        # Replace combobox with numeric sensitivity input
        settings_layout.addWidget(QLabel("Detection Sensitivity (5-30, lower is more sensitive):"))
        self.detection_threshold = QSpinBox()
        self.detection_threshold.setRange(5, 30)
        self.detection_threshold.setValue(16)  # Default medium sensitivity
        self.detection_threshold.setToolTip("Lower values detect more motion but may include noise")
        settings_layout.addWidget(self.detection_threshold)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Group 2: Detection controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        # Motion detection button
        self.btn_run_detection = QPushButton("Run Motion Detection")
        self.btn_run_detection.clicked.connect(self.run_motion_detection)
        controls_layout.addWidget(self.btn_run_detection)
        
        # Abort detection button
        self.btn_abort_detection = QPushButton("Abort Detection")
        self.btn_abort_detection.clicked.connect(self.abort_motion_detection)
        self.btn_abort_detection.setEnabled(False)
        controls_layout.addWidget(self.btn_abort_detection)
        
        # Progress indicator
        self.detection_progress = QProgressBar()
        self.detection_progress.setMinimumWidth(250)
        self.detection_progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        controls_layout.addWidget(self.detection_progress)
        
        # Frame number display
        self.frame_number_label = QLabel("Current frame: -")
        controls_layout.addWidget(self.frame_number_label)
        
        controls_group.setLayout(controls_layout)
        layout.addWidget(controls_group)
        
        # Results summary
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        
        self.detection_summary = QLabel("No detections yet")
        results_layout.addWidget(self.detection_summary)
        
        results_group.setLayout(results_layout)
        layout.addWidget(results_group)
        
        # Add stretch to improve spacing
        layout.addStretch()
        
        widget.setLayout(layout)
        return widget
    
    # Helper methods for directory management
    def get_video_name(self):
        """Get the name of the current video without extension"""
        if not self.video_path:
            return None
        return os.path.splitext(os.path.basename(self.video_path))[0]
    
    def get_annotations_dir_for_current_video(self):
        """Get the annotations directory for the current video"""
        video_name = self.get_video_name()
        if not video_name:
            return None
        video_ann_dir = os.path.join(self.annotations_dir, video_name)
        os.makedirs(video_ann_dir, exist_ok=True)
        return video_ann_dir
    
    def get_frame_path(self, frame_idx, subdir=None):
        """Get path for saving a frame"""
        video_name = self.get_video_name()
        if not video_name:
            return None
            
        if subdir:
            directory = os.path.join(self.datasets_dir, subdir, video_name)
        else:
            directory = os.path.join(self.annotations_dir, video_name)
            
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"frame_{frame_idx:06d}")
    
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
            # Use the new load_video method with source type
            self.video_player.load_video(destination, "Original Video")
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
            QMessageBox.information(self, "Directory Selected", f"Image directory: {directory}")
    
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
    
    def toggle_ui_during_detection(self, is_running):
        """Enable/disable UI components during detection process"""
        self.detection_running = is_running
        
        # Toggle detection buttons
        self.btn_run_detection.setEnabled(not is_running)
        self.btn_abort_detection.setEnabled(is_running)
        self.detection_threshold.setEnabled(not is_running)
        
        # Disable/enable tabs except detection tab
        for i in range(self.tabs.count()):
            if i != 0:  # Detection tab is index 0
                self.tabs.setTabEnabled(i, not is_running)
        
        # Disable/enable menu actions
        for action in self.menuBar().actions():
            action.setEnabled(not is_running)
        
        # Disable ALL video player controls
        if hasattr(self.video_player, 'findChildren'):
            # Disable buttons
            for btn in self.video_player.findChildren(QPushButton):
                btn.setEnabled(not is_running)
            
            # Disable spinboxes
            for spinbox in self.video_player.findChildren(QSpinBox):
                spinbox.setEnabled(not is_running)
            
            # Disable sliders
            for slider in self.video_player.findChildren(QSlider):
                slider.setEnabled(not is_running)
        
        # Update the video info to indicate detection is running
        if is_running:
            self.video_player.set_video_source_type("Detection Preview")
        else:
            self.video_player.set_video_source_type("Original Video")
    
    def run_motion_detection(self):
        if not self.video_path:
            QMessageBox.warning(self, "Missing Input", "Please select a video first.")
            return
        
        # Get and validate the sensitivity value
        sensitivity_value = self.detection_threshold.value()
        
        # Update motion detector settings based on UI input
        self.motion_detector.varThreshold = sensitivity_value
        
        # Clean up any previous detection results
        self.clean_detection_files()
        
        # Disable UI controls during detection
        self.toggle_ui_during_detection(True)
        
        # Start processing thread
        self.process_thread = VideoProcessThread(
            self.video_path, self.motion_detector, self.few_shot
        )
        self.process_thread.update_progress.connect(self.detection_progress.setValue)
        self.process_thread.update_frame.connect(self.update_detection_display)
        self.process_thread.update_frame_number.connect(self.update_frame_number)
        self.process_thread.detection_complete.connect(self.handle_detection_complete)
        
        self.detection_progress.setValue(0)
        self.process_thread.start()
    
    def update_frame_number(self, frame_number):
        """Update the frame number display"""
        self.frame_number_label.setText(f"Current frame: {frame_number}")
    
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
        
        # Re-enable UI controls after detection completes
        self.toggle_ui_during_detection(False)
        
        # Update the UI
        self.detection_summary.setText(f"Detected {len(all_detections)} objects in {len(set([d['frame_idx'] for d in all_detections]))} frames")
        
        # Update the uncertain frames list
        self.uncertain_frames_list.clear()
        for frame_idx, _ in uncertain_frames:
            self.uncertain_frames_list.addItem(f"Frame {frame_idx}")
        
        # Save detection results (bounding boxes as txt instead of images)
        self.save_detection_results()
        
        # Save uncertain frames to the designated directory
        self.save_uncertain_frames()
        
        QMessageBox.information(self, "Detection Complete", 
                              f"Motion detection completed.\n"
                              f"Total detections: {len(all_detections)}\n"
                              f"Uncertain frames: {len(uncertain_frames)}")
    
    def save_detection_results(self):
        if not self.video_path or not self.all_detections:
            return
        
        video_name = self.get_video_name()
        annotations_dir = os.path.join(self.annotations_dir, video_name)
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Group detections by frame
        detections_by_frame = {}
        for detection in self.all_detections:
            frame_idx = detection['frame_idx']
            if frame_idx not in detections_by_frame:
                detections_by_frame[frame_idx] = []
            detections_by_frame[frame_idx].append(detection)
        
        # Save bounding boxes in txt format (one txt file per frame)
        for frame_idx, frame_detections in detections_by_frame.items():
            txt_path = os.path.join(annotations_dir, f"frame_{frame_idx:06d}.txt")
            with open(txt_path, 'w') as f:
                for det in frame_detections:
                    box = det['bbox']
                    label = det['class']
                    conf = det.get('confidence', 0.0)  # Get confidence with fallback
                    # Format: class_name x1 y1 x2 y2 confidence
                    f.write(f"{label} {box[0]} {box[1]} {box[2]} {box[3]} {conf:.4f}\n")
    
    def save_uncertain_frames(self):
        """Save uncertain frames' frame numbers to a text file"""
        if not self.uncertain_frames:
            return
            
        video_name = self.get_video_name()
        os.makedirs(self.uncertain_dir, exist_ok=True)
        
        # Save frame numbers to a text file
        uncertain_txt_path = os.path.join(self.uncertain_dir, f"{video_name}.txt")
        with open(uncertain_txt_path, 'w') as f:
            f.write(f"# Uncertain frames for {video_name}\n")
            f.write(f"# Total uncertain frames: {len(self.uncertain_frames)}\n")
            f.write(f"# Format: frame_index\n")
            
            for frame_idx, _ in self.uncertain_frames:
                f.write(f"{frame_idx}\n")
                
            # Also save detection information for each uncertain frame
            f.write("\n# Detection details for uncertain frames\n")
            f.write("# Format: frame_idx class_name x1 y1 x2 y2 confidence\n")
            
            for frame_idx, _ in self.uncertain_frames:
                frame_dets = [det for det in self.all_detections if det['frame_idx'] == frame_idx]
                for det in frame_dets:
                    box = det['bbox']
                    label = det['class']
                    conf = det.get('confidence', 0.0)
                    f.write(f"{frame_idx} {label} {box[0]} {box[1]} {box[2]} {box[3]} {conf:.4f}\n")
        
        # Update the UI to reflect that we're using a text file instead of images
        self.uncertain_frames_list.clear()
        for frame_idx, _ in self.uncertain_frames:
            self.uncertain_frames_list.addItem(f"Frame {frame_idx}")
    
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
        
        # Save few-shot example
        examples_dir = os.path.join(self.datasets_dir, "few_shot_examples", current_class)
        os.makedirs(examples_dir, exist_ok=True)
        timestamp = os.path.getmtime(self.video_path) if self.video_path else ""
        example_path = os.path.join(examples_dir, f"example_{timestamp}_{len(self.support_examples[current_class])}.jpg")
        cv2.imwrite(example_path, current_frame)
        
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
        video_name = self.get_video_name()
        classified_dir = os.path.join(self.datasets_dir, "classified", video_name)
        os.makedirs(classified_dir, exist_ok=True)
        
        for frame_idx, frame in self.uncertain_frames:
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
            cv2.imwrite(os.path.join(classified_dir, f"frame_{frame_idx:06d}.jpg"), display_frame)
        
        QMessageBox.information(self, "Classification Complete", 
                              "Few-shot classification completed.\nResults saved to classified directory.")
    
    def open_label_tool(self):
        if not self.video_path:
            QMessageBox.warning(self, "No Video Selected", "Please select a video first.")
            return
            
        # Create directory for uncertain frames if it doesn't exist
        video_name = self.get_video_name()
        uncertain_dir = os.path.join(self.datasets_dir, "uncertain", video_name)
        os.makedirs(uncertain_dir, exist_ok=True)
        
        # Save uncertain frames for labeling
        for frame_idx, frame in self.uncertain_frames:
            frame_path = os.path.join(uncertain_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(frame_path, frame)
        
        # Using the separated ImageAnnotatorDialog
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
        if not self.video_path:
            QMessageBox.warning(self, "Warning", "Please select a video first.")
            return False
            
        if len(self.classes) == 0:
            QMessageBox.warning(self, "Warning", "Please define at least one class before preparing training data.")
            return False
            
        # Create training data directory structure
        video_name = self.get_video_name()
        training_dir = os.path.join(self.datasets_dir, "training_data", video_name)
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
                                class_idx = self.classes.index(det['class']) if det.get('class') in self.classes else 0
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
        if not self.video_path:
            QMessageBox.warning(self, "No Video Selected", "Please select a video first.")
            return
        
        video_name = self.get_video_name()
        training_dir = os.path.join(self.datasets_dir, "training_data", video_name)
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
            training_output = os.path.join(self.datasets_dir, "yolo_training", video_name)
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
    
    def show_uncertain_frame(self, item):
        """Display the selected uncertain frame in the video player"""
        # Extract frame index from item text (format: "Frame {frame_idx}")
        try:
            frame_idx = int(item.text().split(' ')[1])
            # Find the matching frame in uncertain_frames list
            for idx, frame_data in self.uncertain_frames:
                if idx == frame_idx:
                    # Display the frame in the video player
                    self.video_player.set_image(frame_data)
                    # Switch to the first tab (video/image display)
                    self.tabs.setCurrentIndex(0)
                    break
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Error", f"Could not display frame: {str(e)}")
    
    def abort_motion_detection(self):
        """Abort the running motion detection process and clean up"""
        if self.process_thread and self.process_thread.isRunning():
            # Stop the processing thread
            self.process_thread.stop()
            self.process_thread.wait()
            
            QMessageBox.information(self, "Detection Aborted", "Motion detection was aborted.")
            
            # Reset progress bar
            self.detection_progress.setValue(0)
            self.frame_number_label.setText("Current frame: -")
            
            # Clean up any created files
            self.clean_detection_files()
            
            # Re-enable UI controls
            self.toggle_ui_during_detection(False)
    
    def clean_detection_files(self):
        """Remove all files created during the detection process"""
        if not self.video_path:
            return
            
        video_name = self.get_video_name()
        
        # Clean annotations directory
        annotations_dir = os.path.join(self.annotations_dir, video_name)
        if os.path.exists(annotations_dir):
            try:
                import shutil
                shutil.rmtree(annotations_dir)
                os.makedirs(annotations_dir, exist_ok=True)
            except Exception as e:
                print(f"Error cleaning annotations directory: {str(e)}")
        
        # Clean uncertain frames file
        uncertain_file = os.path.join(self.uncertain_dir, f"{video_name}.txt")
        if os.path.exists(uncertain_file):
            try:
                os.remove(uncertain_file)
            except Exception as e:
                print(f"Error removing uncertain frames file: {str(e)}")
