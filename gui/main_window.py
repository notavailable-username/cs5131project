from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout, QTabWidget, QPushButton, 
    QFileDialog, QListWidget, QLabel, QMessageBox, QProgressBar, QComboBox,
    QDockWidget, QStackedLayout, QSplitter, QTableWidget, QTableWidgetItem,
    QGroupBox, QLineEdit, QSlider, QSizePolicy
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QPoint
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
import cv2
import os
import numpy as np
import torch
import json

from gui.class_editor_dialog import ClassEditorDialog
from .video_player import VideoPlayer
from .image_annotator_dialog import ImageAnnotatorDialog
from .video_loader_dialog import VideoLoaderDialog  # Import the new dialog
from .yolo_tab import YOLOTab  # Import the new YOLOTab class
from models.motion_detector import MotionDetector
from models.yolo_trainer import YOLOTrainer
import json
import csv

# Add import for FewShotTab
from .few_shot_tab import FewShotTab

class MotionDetectionThread(QThread):
    """Thread dedicated to pure motion detection without few-shot classification"""
    update_progress = pyqtSignal(int)
    update_frame = pyqtSignal(object)
    update_frame_number = pyqtSignal(int)
    detection_complete = pyqtSignal(list)
    
    def __init__(self, video_path, motion_detector, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.motion_detector = motion_detector
        self.running = True
        # Default frame interval calculated in run() method
        self.frame_interval = None
        
    def run(self):
        cap = cv2.VideoCapture(self.video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # If frame_interval wasn't set externally, use the default calculation
        if self.frame_interval is None:
            self.frame_interval = max(1, int(fps / 2))  # Process 2 frames per second by default
        
        all_detections = []
        frame_count = 0
        
        while cap.isOpened() and self.running:
            ret, frame = cap.read()
            if not ret:
                self.update_progress.emit(100)
                break
                
            # Process every nth frame
            if frame_count % self.frame_interval == 0:
                # Update progress and frame number
                progress = min(100, int(100 * frame_count / total_frames))
                self.update_progress.emit(progress)
                self.update_frame_number.emit(frame_count)
                
                # Send frame to GUI for display
                self.update_frame.emit(frame.copy())
                
                # Run motion detection
                boxes = self.motion_detector.detect(frame)
                
                if boxes:
                    frame_h, frame_w = frame.shape[:2]
                    for box in boxes:
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
                        
                        # Record detection in YOLO format with frame_idx, using "-" as class
                        detection = {
                            'frame_idx': frame_count,
                            'bbox_yolo': [center_x, center_y, norm_width, norm_height],
                            'bbox_abs': [x1, y1, x2, y2],  # Keep absolute coords for visualization
                            'class': '-',  # Use dash as placeholder for class
                            'timestamp': frame_count / fps
                        }
                        all_detections.append(detection)
            
            frame_count += 1
            
            if not self.running:
                break
                
        cap.release()
        self.detection_complete.emit(all_detections)
    
    def stop(self):
        self.running = False


class VideoProcessThread(QThread):
    update_progress = pyqtSignal(int)
    update_frame = pyqtSignal(object)
    update_frame_number = pyqtSignal(int)  
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
        
        # Data storage - updated to support multiple videos
        self.current_video_path = None
        self.all_detections = {}  # Dictionary mapping video paths to detections
        self.uncertain_frames = {}  # Dictionary mapping video paths to uncertain frames
        self.support_examples = {}  # {class_name: [image_patches]}
        self.classes = []
        
        # Add video motion settings dictionary
        self.video_motion_settings = {}  # Dictionary mapping video paths to motion detection settings
        
        # Add frame position tracking per video
        self.video_frame_positions = {}  # Dictionary mapping video paths to frame positions
        
        # Batch processing variables
        self.batch_processing = False
        self.video_queue = []
        self.current_batch_index = -1
        
        # Create base directories
        self.datasets_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        self.annotations_dir = os.path.join(self.datasets_dir, "annotations")
        
        # Create annotations/videos directory as well
        self.videos_annotations_dir = os.path.join(self.annotations_dir, "videos")
        os.makedirs(self.videos_annotations_dir, exist_ok=True)
        
        # Create video_configs directory inside annotations dir instead
        self.video_configs_dir = os.path.join(self.datasets_dir, "video_configs")
        os.makedirs(self.video_configs_dir, exist_ok=True)
        
        self.process_thread = None
        self.motion_thread = None
        self.detection_running = False
        
        self._init_ui()
        self._apply_styles()
        
        # Disable controls initially since no videos are loaded
        self.update_controls_state()
    
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
        self.tabs.addTab(self._create_detection_tab(), "Detection")
        self.tabs.addTab(self._create_few_shot_tab(), "Few-Shot Learning")
        # self.tabs.addTab(self._create_annotation_tab(), "Manual Annotation")
        self.tabs.addTab(self._create_training_tab(), "YOLO Training")
        
        # Connect tab changed signal
        self.tabs.currentChanged.connect(self.on_tab_changed)
        
        right_layout.addWidget(self.tabs)
        
        # Add right panel to splitter
        self.main_splitter.addWidget(right_widget)
        
        # Set the initial size ratio
        self.main_splitter.setSizes([600, 600])
        
        # Initialize progress bar to 0%
        if hasattr(self, 'detection_progress'):
            self.detection_progress.setValue(0)
        
        # After creating video_player, connect to its video_changed signal
        self.video_player.video_changed.connect(self.on_video_changed)
    
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
        
        # Text input style for QLineEdit
        text_input_style = """
            QLineEdit {
                color: white;
                background-color: #3d3d3d;
                border: 1px solid #555555;
                border-radius: 2px;
            }
            QLineEdit:hover {
                background-color: #4a4a4a;
            }
            QLineEdit:disabled {
                background-color: #555555;
                color: #888888;
            }
        """
        
        # Apply styles to all buttons in this window
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)
        
        # Apply styles to all text inputs
        for line_edit in self.findChildren(QLineEdit):
            line_edit.setStyleSheet(text_input_style)
            
        for combobox in self.findChildren(QComboBox):
            combobox.setStyleSheet(text_input_style)
    
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
        
        # Add new action for importing multiple videos
        import_multiple_videos_action = file_menu.addAction("Import Multiple Videos")
        import_multiple_videos_action.triggered.connect(self.import_multiple_videos)
        
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
        
        # Edit menu
        edit_menu = menubar.addMenu("Edit")
        
        # Class editor action
        edit_classes_action = edit_menu.addAction("Edit Classes")
        edit_classes_action.triggered.connect(self.open_class_editor)
        
        # Load menu (new)
        load_menu = menubar.addMenu("Load")
        
        # Load videos action
        load_videos_action = load_menu.addAction("Load Videos")
        load_videos_action.triggered.connect(self.open_video_loader)

    def import_video(self):
        """Import a single video file to datasets/videos without loading it to player"""
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
            QMessageBox.information(self, "Video Imported", f"Video imported: {filename}")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import video: {str(e)}")
    
    def import_multiple_videos(self):
        """Import multiple video files to datasets/videos without loading them to player"""
        video_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "", "Video Files (*.mp4 *.avi *.mov)"
        )
        
        if not video_paths:
            return
            
        # Create the datasets/videos directory if it doesn't exist
        videos_dir = os.path.join("datasets", "videos")
        os.makedirs(videos_dir, exist_ok=True)
        
        imported_count = 0
        try:
            import shutil
            for video_path in video_paths:
                filename = os.path.basename(video_path)
                destination = os.path.join(videos_dir, filename)
                shutil.copy2(video_path, destination)
                imported_count += 1
            
            QMessageBox.information(self, "Videos Imported", f"Imported {imported_count} videos")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import videos: {str(e)}")

    def import_videos_directory(self):
        """Import all videos from a directory to datasets/videos without loading them to player"""
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
            
            QMessageBox.information(self, "Videos Imported", f"Imported {imported_count} videos to datasets/videos")
        except Exception as e:
            QMessageBox.critical(self, "Import Error", f"Failed to import videos: {str(e)}")

    def open_video_loader(self):
        """Open the video loader dialog"""
        loader_dialog = VideoLoaderDialog(self)
        loader_dialog.videos_selected.connect(self.on_videos_selected_from_dialog)
        loader_dialog.exec()

    def on_videos_selected_from_dialog(self, video_paths, display_names):
        """Handle videos selection from the video loader dialog"""
        if not video_paths:
            return
            
        # Clear existing videos before adding new ones
        self.video_player.clear_all_videos()
        
        # Add all videos to the player
        for i, video_path in enumerate(video_paths):
            if os.path.exists(video_path):
                self.video_player.add_video(video_path, display_names[i])
        
        # Update current video path
        self.current_video_path = self.video_player.get_current_video_path()
        
        # Load classes from CSV if available
        self.load_classes_from_csv()
        
        # Update controls state
        self.update_controls_state()
        
        QMessageBox.information(self, "Videos Loaded", f"Loaded {len(video_paths)} videos")
        
    def update_controls_state(self):
        """Update the enabled state of controls based on video availability"""
        has_video = self.current_video_path is not None
        has_multiple_videos = hasattr(self, 'video_player') and hasattr(self.video_player, 'videos') and len(self.video_player.videos) > 1
        
        # Update video player controls
        self.video_player.set_controls_enabled(has_video)
        
        # Update detection tab controls
        if hasattr(self, 'btn_run_detection'):
            self.btn_run_detection.setEnabled(has_video)
        if hasattr(self, 'btn_run_all_detection'):
            self.btn_run_all_detection.setEnabled(has_multiple_videos)
        if hasattr(self, 'detection_threshold'):
            self.detection_threshold.setEnabled(has_video)
        if hasattr(self, 'frame_interval'):
            self.frame_interval.setEnabled(has_video)
            
        # Update few-shot tab controls
        if hasattr(self, 'class_selector'):
            self.class_selector.setEnabled(has_video)
            
        # Also disable tabs if needed
        detection_tab_enabled = True  # Always keep detection tab enabled
        few_shot_tab_enabled = has_video
        training_tab_enabled = has_video
            
        self.tabs.setTabEnabled(1, few_shot_tab_enabled)  # Few-Shot tab
        self.tabs.setTabEnabled(2, training_tab_enabled)  # Training tab (YOLO)
    
    def _create_detection_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        # Group 1: Motion detection settings
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        
        # Replace spinbox with text input for sensitivity (removed range limits)
        settings_layout.addWidget(QLabel("Detection Sensitivity (lower values are more sensitive):"))
        self.detection_threshold = QLineEdit()
        self.detection_threshold.setText("25")  # Default medium sensitivity
        self.detection_threshold.setToolTip("Lower values detect more motion but may include noise")
        settings_layout.addWidget(self.detection_threshold)
        
        # Add frame interval setting
        settings_layout.addWidget(QLabel("Frame Interval (frames per second to process):"))
        self.frame_interval = QLineEdit()
        self.frame_interval.setText("2")  # Default 2 fps processing
        self.frame_interval.setToolTip("Higher values process more frames but take longer")
        settings_layout.addWidget(self.frame_interval)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # Group 2: Detection controls
        controls_group = QGroupBox("Controls")
        controls_layout = QVBoxLayout()
        
        # Motion detection button
        self.btn_run_detection = QPushButton("Run Motion Detection")
        self.btn_run_detection.clicked.connect(self.run_motion_detection)
        controls_layout.addWidget(self.btn_run_detection)
        
        # Add new button for running detection on all videos
        self.btn_run_all_detection = QPushButton("Run Motion Detection on All Videos")
        self.btn_run_all_detection.clicked.connect(self.run_motion_detection_all_videos)
        controls_layout.addWidget(self.btn_run_all_detection)
        
        # Abort detection button
        self.btn_abort_detection = QPushButton("Abort Detection")
        self.btn_abort_detection.clicked.connect(self.abort_motion_detection)
        self.btn_abort_detection.setEnabled(False)
        controls_layout.addWidget(self.btn_abort_detection)
        
        # Progress indicator
        self.detection_progress = QProgressBar()
        self.detection_progress.setMinimumWidth(250)
        self.detection_progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.detection_progress.setValue(0)  # Initialize to 0%
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
        if not self.current_video_path:
            return None
        return os.path.splitext(os.path.basename(self.current_video_path))[0]
    
    def get_annotations_dir_for_current_video(self):
        """Get the annotations directory for the current video"""
        video_name = self.get_video_name()
        if not video_name:
            return None
        # Updated to include "videos" in the path
        video_ann_dir = os.path.join(self.annotations_dir, "videos", video_name)
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
            # Updated to include "videos" in the path for annotations
            directory = os.path.join(self.annotations_dir, "videos", video_name)
            
        os.makedirs(directory, exist_ok=True)
        return os.path.join(directory, f"frame_{frame_idx:06d}")
    
    def load_classes_from_csv(self):
        """Load class definitions from classes.csv in the video's annotation directory"""
        if not self.current_video_path:
            return
            
        video_name = self.get_video_name()
        # Updated to include "videos" in the path
        classes_csv_path = os.path.join(self.annotations_dir, "videos", video_name, "classes.csv")
        
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
                
                # Update class selector in the Few-Shot tab
                if hasattr(self, 'few_shot_tab'):
                    self.few_shot_tab.update_classes(self.classes)
                    
            except Exception as e:
                print(f"Error loading classes from CSV: {str(e)}")
        else:
            # If no CSV exists, just have empty classes
            pass
            
        # Update UI with loaded classes
        if hasattr(self, 'class_selector'):
            self.class_selector.clear()
            if self.classes:
                self.class_selector.addItems(self.classes)
    
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
        """Creates the few-shot learning tab by instantiating the FewShotTab class"""
        self.few_shot_tab = FewShotTab(self)
        
        # Connect signals
        self.few_shot_tab.request_frame_seek.connect(self.video_player.seek)
        self.few_shot_tab.annotation_saved.connect(self.on_annotation_saved)
        
        # Initialize with current classes if available
        if hasattr(self, 'classes') and self.classes:
            self.few_shot_tab.update_classes(self.classes)
            
        return self.few_shot_tab

    def on_annotation_saved(self, frame_idx, annotations):
        """Handle when annotations are saved in the few-shot tab"""
        # Placeholder for additional logic when annotations are saved
        pass

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
        """Creates the YOLO training tab by instantiating the YOLOTab class"""
        return YOLOTab(self)
    
    def load_images(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if directory:
            QMessageBox.information(self, "Directory Selected", f"Image directory: {directory}")
    
    def toggle_ui_during_detection(self, is_running):
        """Enable/disable UI components during detection process"""
        self.detection_running = is_running
        
        # Toggle detection buttons
        self.btn_run_detection.setEnabled(not is_running)
        self.btn_run_all_detection.setEnabled(not is_running)
        self.btn_abort_detection.setEnabled(is_running)
        self.detection_threshold.setEnabled(not is_running)
        self.frame_interval.setEnabled(not is_running)  # Also disable frame interval input
        
        # Disable/enable tabs except detection tab
        for i in range(self.tabs.count()):
            if i != 0:  # Detection tab is index 0
                self.tabs.setTabEnabled(i, not is_running)
        
        # Disable/enable menu actions
        for action in self.menuBar().actions():
            action.setEnabled(not is_running)
        
        # Disable video player controls during detection
        self.video_player.set_controls_enabled(not is_running)
        
        # Update the video info to indicate detection is running
        if is_running:
            self.video_player.set_video_source_type("Detection Preview")
        else:
            # Use the actual filename of the current video
            if self.current_video_path:
                current_filename = os.path.basename(self.current_video_path)
                self.video_player.set_video_source_type(current_filename)
            else:
                self.video_player.set_video_source_type("No Video")
    
    def run_motion_detection(self):
        if not self.current_video_path:
            QMessageBox.warning(self, "Missing Input", "Please select a video first.")
            return
        
        # Save current frame position before starting detection
        self.video_frame_positions[self.current_video_path] = self.video_player.get_current_frame_idx()
        
        # Save current settings from UI to memory and to file
        self.update_settings_from_ui()
        self.save_video_motion_settings(self.current_video_path)
        
        # Get settings for detection
        settings = self.video_motion_settings[self.current_video_path]["motion_detection_settings"]
        sensitivity_value = settings["sensitivity"]
        frame_interval = settings["frame_interval"]
        
        # Clean up any previous detection results for this video
        self.clean_detection_files()
        
        # Disable UI controls during detection
        self.toggle_ui_during_detection(True)
        
        # Create a fresh MotionDetector instance with the correct settings
        md_conf = self.config.get("motion_detector", {}).copy()
        md_conf["varThreshold"] = sensitivity_value
        video_motion_detector = MotionDetector(**md_conf)
        
        # Start motion detection thread with the specific detector for this video
        self.motion_thread = MotionDetectionThread(
            self.current_video_path, video_motion_detector
        )
        
        # Override the thread's frame_interval with user input
        cap = cv2.VideoCapture(self.current_video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        self.motion_thread.frame_interval = max(1, int(fps / frame_interval))
        
        self.motion_thread.update_progress.connect(self.detection_progress.setValue)
        self.motion_thread.update_frame.connect(self.update_detection_display)
        self.motion_thread.update_frame_number.connect(self.update_frame_number)
        self.motion_thread.detection_complete.connect(self.handle_motion_detection_complete)
        
        self.detection_progress.setValue(0)
        self.motion_thread.start()
    
    def run_motion_detection_all_videos(self):
        """Run motion detection on all loaded videos sequentially"""
        if not hasattr(self.video_player, 'videos') or len(self.video_player.videos) == 0:
            QMessageBox.warning(self, "Missing Input", "No videos loaded.")
            return
        
        # Save current video position before starting batch detection
        if self.current_video_path:
            self.video_frame_positions[self.current_video_path] = self.video_player.get_current_frame_idx()
        
        # Save current settings from UI before starting batch processing
        if self.current_video_path:
            self.update_settings_from_ui()
            self.save_video_motion_settings(self.current_video_path)
        
        # Initialize batch processing
        self.batch_processing = True
        self.video_queue = [video['path'] for video in self.video_player.videos]
        self.current_batch_index = -1
        
        # Start with the first video
        self.process_next_video_in_queue()
        
    def process_next_video_in_queue(self):
        """Process the next video in the queue for batch detection"""
        if not self.batch_processing or not self.video_queue:
            # Batch processing complete or aborted
            self.batch_processing = False
            self.toggle_ui_during_detection(False)
            QMessageBox.information(self, "Batch Processing Complete", 
                                  "Motion detection completed on all videos.")
            return
            
        # Move to next video in queue
        self.current_batch_index += 1
        if self.current_batch_index >= len(self.video_queue):
            # All videos processed
            self.batch_processing = False
            self.toggle_ui_during_detection(False)
            QMessageBox.information(self, "Batch Processing Complete", 
                                  "Motion detection completed on all videos.")
            return
        
        # Get the next video path
        next_video_path = self.video_queue[self.current_batch_index]
        
        # Find the index in video_player's videos list
        next_video_index = -1
        for i, video in enumerate(self.video_player.videos):
            if video['path'] == next_video_path:
                next_video_index = i
                break
        
        if next_video_index >= 0:
            # Switch to this video
            self.video_player.switch_to_video(next_video_index)
            self.current_video_path = next_video_path
            
            # Load settings for this video
            settings = self.load_video_motion_settings(next_video_path)
            md_settings = settings["motion_detection_settings"]
            sensitivity_value = md_settings["sensitivity"]
            frame_interval_value = md_settings["frame_interval"]
            
            # Update UI with these settings
            if hasattr(self, 'detection_threshold'):
                self.detection_threshold.setText(str(sensitivity_value))
            if hasattr(self, 'frame_interval'):
                self.frame_interval.setText(str(frame_interval_value))
            
            # Update UI to show current video being processed
            video_name = os.path.basename(next_video_path)
            self.detection_summary.setText(f"Processing video {self.current_batch_index + 1} of {len(self.video_queue)}: {video_name}")
            
            # Clean detection files for this video
            self.clean_detection_files()
            
            # Disable UI controls during detection
            self.toggle_ui_during_detection(True)
            
            # Create a fresh MotionDetector instance with the correct settings for this video
            # Get the base configuration from the main config
            md_conf = self.config.get("motion_detector", {}).copy()
            # Override with video-specific settings
            md_conf["varThreshold"] = sensitivity_value
            # Create a new instance with these settings
            video_motion_detector = MotionDetector(**md_conf)
            
            # Start motion detection with the video-specific detector
            self.motion_thread = MotionDetectionThread(
                self.current_video_path, video_motion_detector
            )
            
            # Set frame interval for the thread
            cap = cv2.VideoCapture(self.current_video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            self.motion_thread.frame_interval = max(1, int(fps / frame_interval_value))
            
            # Connect signals
            self.motion_thread.update_progress.connect(self.detection_progress.setValue)
            self.motion_thread.update_frame.connect(self.update_detection_display)
            self.motion_thread.update_frame_number.connect(self.update_frame_number)
            self.motion_thread.detection_complete.connect(self.handle_motion_detection_complete)
            
            # Reset progress bar
            self.detection_progress.setValue(0)
            
            # Start detection
            self.motion_thread.start()
        else:
            # Skip to next if this video is no longer in the player
            self.process_next_video_in_queue()
    
    def update_frame_number(self, frame_number):
        """Update the frame number display"""
        self.frame_number_label.setText(f"Current frame: {frame_number}")
    
    def update_detection_display(self, frame):
        """Update the display with the current detection frame"""
        self.current_frame = frame
        
        # Get the current motion detector being used by the active thread
        current_detector = None
        if self.motion_thread and self.motion_thread.isRunning():
            current_detector = self.motion_thread.motion_detector
        else:
            # Fall back to default detector if no thread is running
            current_detector = self.motion_detector
        
        # Draw bounding boxes from motion detector
        boxes = current_detector.detect(frame)
        frame_annotations = []
        
        for box in boxes:
            x1, y1, x2, y2 = box
            # Create annotation dict for visualization - no class or confidence
            annotation = {
                'bbox_abs': [x1, y1, x2, y2]
                # No class or confidence information
            }
            frame_annotations.append(annotation)
        
        # Use the video player to display the frame with annotations, but don't show labels
        self.video_player.set_image_with_annotations(frame, frame_annotations, show_labels=False)
    
    def handle_motion_detection_complete(self, all_detections):
        """Handle the completion of motion detection without few-shot classification"""
        # Store the detections for the current video
        self.all_detections[self.current_video_path] = all_detections
        
        # Re-enable UI controls
        self.toggle_ui_during_detection(False)
        
        # Update the summary
        unique_frames = len(set([d['frame_idx'] for d in all_detections]))
        self.detection_summary.setText(f"Detected {len(all_detections)} objects in {unique_frames} frames")
        
        # Save detection results in YOLO format
        self.save_motion_detection_results()
        
        # Restore the player to the original frame position using the per-video tracking
        if self.current_video_path in self.video_frame_positions:
            self.video_player.seek(self.video_frame_positions[self.current_video_path])
        
        # If we're in batch mode, continue with the next video
        if self.batch_processing:
            # Small delay to allow UI to update
            QTimer.singleShot(500, self.process_next_video_in_queue)
        else:
            # Otherwise, re-enable UI controls
            self.toggle_ui_during_detection(False)
            
            # Restore the player to the original frame position using the per-video tracking
            if self.current_video_path in self.video_frame_positions:
                self.video_player.seek(max(0, self.video_frame_positions[self.current_video_path]))
            
            QMessageBox.information(self, "Detection Complete", 
                                  f"Motion detection completed.\n"
                                  f"Total detections: {len(all_detections)}")
        
        # Reset progress bar to 0% when detection is complete for this video
        self.detection_progress.setValue(0)
    
    def save_motion_detection_results(self):
        """Save motion detection results in standardized YOLO format"""
        if not self.current_video_path or self.current_video_path not in self.all_detections:
            return
        
        all_detections = self.all_detections[self.current_video_path]
        
        video_name = self.get_video_name()
        # Updated to include "videos" in the path
        annotations_dir = os.path.join(self.annotations_dir, "videos", video_name)
        os.makedirs(annotations_dir, exist_ok=True)
        
        # Group detections by frame
        detections_by_frame = {}
        for detection in all_detections:
            frame_idx = detection['frame_idx']
            if frame_idx not in detections_by_frame:
                detections_by_frame[frame_idx] = []
            detections_by_frame[frame_idx].append(detection)
        
        # Save bounding boxes in YOLO format (one txt file per frame)
        for frame_idx, frame_detections in detections_by_frame.items():
            txt_path = os.path.join(annotations_dir, f"frame_{frame_idx:06d}.txt")
            with open(txt_path, 'w') as f:
                for det in frame_detections:
                    # Use the YOLO bbox format (center_x center_y width height)
                    bbox = det['bbox_yolo']
                    # Use dash as placeholder for class
                    f.write(f"- {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")
        
        # Create the classes.csv file only if it doesn't exist
        classes_csv_path = os.path.join(annotations_dir, "classes.csv")
        if not os.path.exists(classes_csv_path):
            with open(classes_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['class_id', 'class_name'])
                # Only write actual defined classes, no default classes
                for i, class_name in enumerate(self.classes):
                    writer.writerow([i, class_name])

    def clean_detection_files(self):
        """Remove all files created during the detection process"""
        if not self.current_video_path:
            return
            
        video_name = self.get_video_name()
        
        # Clean annotations directory - updated to include "videos" in the path
        annotations_dir = os.path.join(self.annotations_dir, "videos", video_name)
        if os.path.exists(annotations_dir):
            try:
                import shutil
                shutil.rmtree(annotations_dir)
                os.makedirs(annotations_dir, exist_ok=True)
            except Exception as e:
                print(f"Error cleaning annotations directory: {str(e)}")
                
        # Clear any stored detections for this video
        if self.current_video_path in self.all_detections:
            del self.all_detections[self.current_video_path]
            
        # Clear any stored uncertain frames for this video
        if self.current_video_path in self.uncertain_frames:
            del self.uncertain_frames[self.current_video_path]
            
        # Clear the uncertain frames list UI
        self.update_uncertain_frames_list()
    
    def show_uncertain_frame(self, item):
        """Display the selected uncertain frame in the video player"""
        try:
            frame_idx = int(item.text().split(' ')[1])
            
            # Load the original video and seek to the frame
            if self.current_video_path and os.path.exists(self.current_video_path):
                # Use video_player to seek to the frame
                self.video_player.seek(frame_idx)
                
                # Switch to the first tab
                self.tabs.setCurrentIndex(0)
        except (ValueError, IndexError) as e:
            QMessageBox.warning(self, "Error", f"Could not display frame: {str(e)}")
    
    def abort_motion_detection(self):
        """Abort any running motion detection process"""
        if self.motion_thread and self.motion_thread.isRunning():
            # Stop the motion detection thread
            self.motion_thread.stop()
            self.motion_thread.wait()
            
            message = "Motion detection was aborted."
            if self.batch_processing:
                message += " Batch processing canceled."
                self.batch_processing = False
            
            QMessageBox.information(self, "Detection Aborted", message)
            
        elif self.process_thread and self.process_thread.isRunning():
            # Stop the video processing thread if that's running instead
            self.process_thread.stop()
            self.process_thread.wait()
            
            message = "Video processing was aborted."
            if self.batch_processing:
                message += " Batch processing canceled."
                self.batch_processing = False
            
            QMessageBox.information(self, "Detection Aborted", message)
        
        # Reset progress and UI regardless of which thread was running
        self.detection_progress.setValue(0)
        self.frame_number_label.setText("Current frame: -")
        
        # Clean up detection files
        self.clean_detection_files()
        
        # Re-enable UI controls
        self.toggle_ui_during_detection(False)
        
        # Restore the player to the original frame position using the per-video tracking
        if self.current_video_path in self.video_frame_positions:
            self.video_player.seek(self.video_frame_positions[self.current_video_path])
    
    def open_label_tool(self):
        if not self.current_video_path:
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

    def update_uncertain_frames_list(self):
        """Update the uncertain frames list for the current video"""
        if hasattr(self, 'uncertain_frames_list'):
            self.uncertain_frames_list.clear()
            
            if self.current_video_path in self.uncertain_frames:
                uncertain_frames = self.uncertain_frames[self.current_video_path]
                for frame_idx, _ in uncertain_frames:
                    self.uncertain_frames_list.addItem(f"Frame {frame_idx}")
    
    def on_video_changed(self, index):
        """Handle video change in the player"""
        # If there was a previous video, save its settings
        if self.current_video_path:
            self.update_settings_from_ui()
            self.save_video_motion_settings(self.current_video_path)
        
        # Update current video path
        old_path = self.current_video_path
        self.current_video_path = self.video_player.get_current_video_path()
        
        # Only proceed if the path actually changed
        if old_path != self.current_video_path:
            # Load settings for new video and update UI
            self.update_ui_from_settings()
            
            # Load classes for this video
            self.load_classes_from_csv()
            
            # Clear detection progress and frame labels
            if hasattr(self, 'detection_progress'):
                self.detection_progress.setValue(0)
            if hasattr(self, 'frame_number_label'):
                self.frame_number_label.setText("Current frame: -")
                
            # Update detection summary if we have results for this video
            if self.current_video_path in self.all_detections:
                detections = self.all_detections[self.current_video_path]
                unique_frames = len(set([d['frame_idx'] for d in detections]))
                self.detection_summary.setText(f"Detected {len(detections)} objects in {unique_frames} frames")
            else:
                self.detection_summary.setText("No detections yet")
                
            # Update uncertain frames list if we have those for this video
            self.update_uncertain_frames_list()
            
            # If we're currently on the few-shot tab, update its contents
            if self.tabs.currentIndex() == 1 and hasattr(self, 'few_shot_tab'):
                current_frame_idx = self.video_player.get_current_frame_idx()
                self.few_shot_tab.load_and_display_annotations(current_frame_idx)
        
        # Update controls state
        self.update_controls_state()
        
        if hasattr(self, 'few_shot_tab'):
            self.few_shot_tab.clear_annotations()
    
    def open_class_editor(self):
        """Open the class editor dialog"""
        if not self.video_player or not hasattr(self.video_player, 'videos') or not self.video_player.videos:
            QMessageBox.warning(self, "No Videos", "Please load at least one video before editing classes.")
            return
        
        # Get the list of videos from the video player
        videos = self.video_player.videos
        
        # Create and show the class editor dialog
        dialog = ClassEditorDialog(self, self.annotations_dir, videos)
        dialog.classes_updated.connect(self.on_classes_updated)
        
        # Show dialog as modal
        dialog.exec()

    def on_classes_updated(self, video_path, class_names):
        """Handle when classes are updated in the editor"""
        # Update the classes list for the current video if it matches
        if self.current_video_path == video_path:
            self.classes = class_names
            
            # Update class selector in the Few-Shot tab if it exists
            if hasattr(self, 'class_selector'):
                self.class_selector.clear()
                self.class_selector.addItems(self.classes)
                
        if hasattr(self, 'few_shot_tab'):
            self.few_shot_tab.update_classes(class_names)
    
    def load_video_motion_settings(self, video_path):
        """Load motion detection settings for a specific video from JSON file"""
        if not video_path:
            return None
            
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        config_file = os.path.join(self.video_configs_dir, f"{video_name}.json")
        
        default_settings = {
            "motion_detection_settings": {
                "sensitivity": 25,  # Default sensitivity
                "frame_interval": 2  # Default frame interval (fps to process)
            }
        }
        
        # If settings already loaded in memory, return those
        if video_path in self.video_motion_settings:
            return self.video_motion_settings[video_path]
            
        # Otherwise try to load from file
        if os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    settings = json.load(f)
                # Cache the settings in memory
                self.video_motion_settings[video_path] = settings
                return settings
            except Exception as e:
                print(f"Error loading motion settings for {video_name}: {e}")
        
        # If file doesn't exist or has errors, use defaults and save them
        self.video_motion_settings[video_path] = default_settings
        self.save_video_motion_settings(video_path)
        
        return default_settings
    
    def save_video_motion_settings(self, video_path):
        """Save motion detection settings for a specific video to JSON file"""
        if not video_path or video_path not in self.video_motion_settings:
            return
        
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        config_file = os.path.join(self.video_configs_dir, f"{video_name}.json")
        
        try:
            with open(config_file, 'w') as f:
                json.dump(self.video_motion_settings[video_path], f, indent=4)
        except Exception as e:
            print(f"Error saving motion settings for {video_name}: {e}")
    
    def update_ui_from_settings(self):
        """Update UI controls based on loaded settings for the current video"""
        if not self.current_video_path:
            return
            
        settings = self.load_video_motion_settings(self.current_video_path)
        if not settings:
            return
            
        md_settings = settings.get("motion_detection_settings", {})
        
        # Update sensitivity input
        if hasattr(self, 'detection_threshold') and 'sensitivity' in md_settings:
            self.detection_threshold.setText(str(md_settings['sensitivity']))
            
        # Update frame interval input
        if hasattr(self, 'frame_interval') and 'frame_interval' in md_settings:
            self.frame_interval.setText(str(md_settings['frame_interval']))
    
    def update_settings_from_ui(self):
        """Update settings dictionary from UI controls for current video"""
        if not self.current_video_path:
            return
            
        # Get current settings or create new
        if self.current_video_path not in self.video_motion_settings:
            self.video_motion_settings[self.current_video_path] = {
                "motion_detection_settings": {}
            }
        
        md_settings = self.video_motion_settings[self.current_video_path]["motion_detection_settings"]
        
        # Update from UI values
        try:
            md_settings["sensitivity"] = int(self.detection_threshold.text())
        except (ValueError, AttributeError):
            md_settings["sensitivity"] = 25  # Default if invalid
            
        try:
            md_settings["frame_interval"] = int(self.frame_interval.text())
        except (ValueError, AttributeError):
            md_settings["frame_interval"] = 2  # Default if invalid
    
    def on_tab_changed(self, index):
        """Handle tab change events"""
        if index == 1:  # Few-Shot Learning tab index
            # Update few-shot tab with current classes
            if hasattr(self, 'few_shot_tab') and self.classes:
                self.few_shot_tab.update_classes(self.classes)
        elif index == 2:  # YOLO Training tab
            # Ensure YOLO tab has current classes if available
            yolo_tab = self.tabs.widget(2)
            if isinstance(yolo_tab, YOLOTab) and hasattr(self, 'classes') and self.classes:
                # Method to update classes would be added to YOLOTab if needed
                pass
