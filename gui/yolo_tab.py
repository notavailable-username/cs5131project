from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QComboBox,
    QMessageBox, QGroupBox, QFileDialog, QProgressBar, QSpinBox,
    QCheckBox, QSplitter, QFormLayout, QTextEdit, QSlider, QDialog,
    QDoubleSpinBox, QGridLayout, QProgressDialog
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread
import os
import cv2
import json
import numpy as np
import shutil
from models.yolo_trainer import YOLOTrainer  # Import the actual trainer

class AugmentationSettingsDialog(QDialog):
    """Dialog for configuring data augmentation settings"""
    
    def __init__(self, parent=None, current_settings=None):
        super().__init__(parent)
        self.setWindowTitle("Data Augmentation Settings")
        self.resize(400, 500)
        
        # Default settings
        self.settings = current_settings or {
            "mosaic": 1.0,
            "mixup": 0.1,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "random_crop": False
        }
        
        self._init_ui()
        self._load_settings()
    
    def _init_ui(self):
        layout = QVBoxLayout(self)
        
        # Create a grid for the settings
        form = QGridLayout()
        row = 0
        
        # Add controls for each augmentation setting with tooltips
        
        # Mosaic
        form.addWidget(QLabel("Mosaic:"), row, 0)
        self.mosaic = QDoubleSpinBox()
        self.mosaic.setRange(0.0, 1.0)
        self.mosaic.setSingleStep(0.1)
        self.mosaic.setToolTip("Mosaic augmentation, combining 4 images. 0=disabled, 1=full effect")
        form.addWidget(self.mosaic, row, 1)
        row += 1
        
        # Mixup
        form.addWidget(QLabel("Mixup:"), row, 0)
        self.mixup = QDoubleSpinBox()
        self.mixup.setRange(0.0, 1.0)
        self.mixup.setSingleStep(0.1)
        self.mixup.setToolTip("Mixup augmentation, blending 2 images. 0=disabled, 1=full effect")
        form.addWidget(self.mixup, row, 1)
        row += 1
        
        # Degrees
        form.addWidget(QLabel("Rotation (degrees):"), row, 0)
        self.degrees = QDoubleSpinBox()
        self.degrees.setRange(0.0, 45.0)
        self.degrees.setSingleStep(1.0)
        self.degrees.setToolTip("Maximum rotation degrees. 0=disabled")
        form.addWidget(self.degrees, row, 1)
        row += 1
        
        # Translate
        form.addWidget(QLabel("Translate:"), row, 0)
        self.translate = QDoubleSpinBox()
        self.translate.setRange(0.0, 1.0)
        self.translate.setSingleStep(0.05)
        self.translate.setToolTip("Translation ratio. 0=disabled, 1=full image size")
        form.addWidget(self.translate, row, 1)
        row += 1
        
        # Scale
        form.addWidget(QLabel("Scale:"), row, 0)
        self.scale = QDoubleSpinBox()
        self.scale.setRange(0.0, 1.0)
        self.scale.setSingleStep(0.05)
        self.scale.setToolTip("Scale images up and down. 0=no scaling")
        form.addWidget(self.scale, row, 1)
        row += 1
        
        # Shear
        form.addWidget(QLabel("Shear:"), row, 0)
        self.shear = QDoubleSpinBox()
        self.shear.setRange(0.0, 1.0)
        self.shear.setSingleStep(0.05)
        self.shear.setToolTip("Shear image. 0=disabled")
        form.addWidget(self.shear, row, 1)
        row += 1
        
        # Perspective
        form.addWidget(QLabel("Perspective:"), row, 0)
        self.perspective = QDoubleSpinBox()
        self.perspective.setRange(0.0, 0.001)
        self.perspective.setSingleStep(0.0001)
        self.perspective.setDecimals(4)
        self.perspective.setToolTip("Perspective transformation. 0=disabled")
        form.addWidget(self.perspective, row, 1)
        row += 1
        
        # Flip Up/Down
        form.addWidget(QLabel("Flip Up/Down:"), row, 0)
        self.flipud = QDoubleSpinBox()
        self.flipud.setRange(0.0, 1.0)
        self.flipud.setSingleStep(0.1)
        self.flipud.setToolTip("Flip image vertically. 0=disabled, 0.5=50% of images will be flipped")
        form.addWidget(self.flipud, row, 1)
        row += 1
        
        # Flip Left/Right
        form.addWidget(QLabel("Flip Left/Right:"), row, 0)
        self.fliplr = QDoubleSpinBox()
        self.fliplr.setRange(0.0, 1.0)
        self.fliplr.setSingleStep(0.1)
        self.fliplr.setToolTip("Flip image horizontally. 0=disabled, 0.5=50% of images will be flipped")
        form.addWidget(self.fliplr, row, 1)
        row += 1
        
        # HSV Hue
        form.addWidget(QLabel("HSV Hue:"), row, 0)
        self.hsv_h = QDoubleSpinBox()
        self.hsv_h.setRange(0.0, 0.1)
        self.hsv_h.setSingleStep(0.01)
        self.hsv_h.setDecimals(3)
        self.hsv_h.setToolTip("HSV-Hue augmentation. 0=disabled")
        form.addWidget(self.hsv_h, row, 1)
        row += 1
        
        # HSV Saturation
        form.addWidget(QLabel("HSV Saturation:"), row, 0)
        self.hsv_s = QDoubleSpinBox()
        self.hsv_s.setRange(0.0, 1.0)
        self.hsv_s.setSingleStep(0.1)
        self.hsv_s.setToolTip("HSV-Saturation augmentation. 0=disabled")
        form.addWidget(self.hsv_s, row, 1)
        row += 1
        
        # HSV Value
        form.addWidget(QLabel("HSV Value:"), row, 0)
        self.hsv_v = QDoubleSpinBox()
        self.hsv_v.setRange(0.0, 1.0)
        self.hsv_v.setSingleStep(0.1)
        self.hsv_v.setToolTip("HSV-Value augmentation. 0=disabled")
        form.addWidget(self.hsv_v, row, 1)
        row += 1
        
        # Random cropping
        form.addWidget(QLabel("Random Cropping:"), row, 0)
        self.random_crop = QCheckBox()
        self.random_crop.setToolTip("Enable random cropping to generate more training data. This will disable aspect ratio preservation.")
        form.addWidget(self.random_crop, row, 1)
        row += 1
        
        # Add form to layout
        layout.addLayout(form)
        
        # Add reset button
        reset_button = QPushButton("Reset to Defaults")
        reset_button.clicked.connect(self._reset_defaults)
        layout.addWidget(reset_button)
        
        # Add buttons
        button_box = QHBoxLayout()
        self.ok_button = QPushButton("OK")
        self.ok_button.clicked.connect(self.accept)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        
        button_box.addWidget(self.cancel_button)
        button_box.addWidget(self.ok_button)
        layout.addLayout(button_box)
    
    def _load_settings(self):
        """Load settings into the UI controls"""
        self.mosaic.setValue(self.settings.get("mosaic", 1.0))
        self.mixup.setValue(self.settings.get("mixup", 0.1))
        self.degrees.setValue(self.settings.get("degrees", 0.0))
        self.translate.setValue(self.settings.get("translate", 0.1))
        self.scale.setValue(self.settings.get("scale", 0.5))
        self.shear.setValue(self.settings.get("shear", 0.0))
        self.perspective.setValue(self.settings.get("perspective", 0.0))
        self.flipud.setValue(self.settings.get("flipud", 0.0))
        self.fliplr.setValue(self.settings.get("fliplr", 0.5))
        self.hsv_h.setValue(self.settings.get("hsv_h", 0.015))
        self.hsv_s.setValue(self.settings.get("hsv_s", 0.7))
        self.hsv_v.setValue(self.settings.get("hsv_v", 0.4))
        self.random_crop.setChecked(self.settings.get("random_crop", False))
    
    def _reset_defaults(self):
        """Reset to default values"""
        self.settings = {
            "mosaic": 1.0,
            "mixup": 0.1,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "random_crop": False
        }
        self._load_settings()
    
    def get_settings(self):
        """Get the current settings from the dialog"""
        return {
            "mosaic": self.mosaic.value(),
            "mixup": self.mixup.value(),
            "degrees": self.degrees.value(),
            "translate": self.translate.value(),
            "scale": self.scale.value(),
            "shear": self.shear.value(),
            "perspective": self.perspective.value(),
            "flipud": self.flipud.value(),
            "fliplr": self.fliplr.value(),
            "hsv_h": self.hsv_h.value(),
            "hsv_s": self.hsv_s.value(),
            "hsv_v": self.hsv_v.value(),
            "random_crop": self.random_crop.isChecked()
        }

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
            self.log_update.emit(f"Training with {self.config['epochs']} epochs...")
            self.log_update.emit(f"Model: {self.config['model_type']}")
            self.log_update.emit(f"Image size: {self.config['img_size']}px")
            self.log_update.emit(f"Batch size: {self.config['batch_size']}")
            
            # Log augmentation settings if enabled
            if self.config.get('augmentation', False):
                self.log_update.emit("Data augmentation enabled with settings:")
                for key in ['mosaic', 'mixup', 'degrees', 'translate', 'scale', 'shear', 
                           'perspective', 'flipud', 'fliplr', 'hsv_h', 'hsv_s', 'hsv_v']:
                    if key in self.config:
                        self.log_update.emit(f"  - {key}: {self.config[key]}")
                        
                if self.config.get('random_crop', False):
                    self.log_update.emit("  - Random cropping enabled")
            else:
                self.log_update.emit("Data augmentation disabled")
            
            # Use the actual trainer to run training (instead of simulation)
            results = self.trainer.train()
            
            if results['success']:
                self.log_update.emit(f"Training completed successfully!")
                self.log_update.emit(f"Model saved to: {results.get('model_path', 'unknown')}")
                if 'mAP' in results:
                    self.log_update.emit(f"mAP: {results['mAP']:.4f}")
                self.training_complete.emit(True, "Training completed successfully")
            else:
                self.log_update.emit(f"Training failed: {results.get('error', 'unknown error')}")
                self.training_complete.emit(False, f"Training failed: {results.get('error', 'unknown error')}")
            
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
        
        # Initialize augmentation settings
        self.augmentation_settings = {
            "mosaic": 1.0,
            "mixup": 0.1,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "random_crop": False
        }
        
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
        
        # Model selection - Updated with all YOLOv9, YOLOv10, and YOLOv11 variants
        self.model_combo = QComboBox()
        # YOLOv11 variants
        self.model_combo.addItems([
            "YOLOv11n", "YOLOv11s", "YOLOv11m", "YOLOv11l", "YOLOv11x"
        ])
        # YOLOv10 variants
        self.model_combo.addItems([
            "YOLOv10n", "YOLOv10s", "YOLOv10m", "YOLOv10l", "YOLOv10x"
        ])
        # YOLOv9 variants
        self.model_combo.addItems([
            "YOLOv9c", "YOLOv9e", "YOLOv9n", "YOLOv9s", "YOLOv9m", "YOLOv9l", "YOLOv9x"
        ])
        # YOLOv8 variants
        self.model_combo.addItems([
            "YOLOv8n", "YOLOv8s", "YOLOv8m", "YOLOv8l", "YOLOv8x"
        ])
        
        self.model_combo.setCurrentIndex(0)  # Set YOLOv11n as default
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
        
        # Augmentation options with layout
        aug_layout = QHBoxLayout()
        self.augmentation_check = QCheckBox("Enable Data Augmentation")
        self.augmentation_check.setChecked(True)
        self.augmentation_check.setToolTip("Apply data augmentation to increase dataset variety")
        self.augmentation_check.stateChanged.connect(self.toggle_augmentation)
        aug_layout.addWidget(self.augmentation_check)
        
        # Add augmentation settings button
        self.btn_aug_settings = QPushButton("Settings")
        self.btn_aug_settings.clicked.connect(self.open_augmentation_settings)
        self.btn_aug_settings.setToolTip("Configure data augmentation settings")
        self.btn_aug_settings.setEnabled(self.augmentation_check.isChecked())
        aug_layout.addWidget(self.btn_aug_settings)
        
        settings_layout.addRow(aug_layout)
        
        # Add aspect ratio option and random crop option
        aspect_crop_layout = QHBoxLayout()
        
        # Preserve aspect ratio checkbox
        self.preserve_aspect_ratio = QCheckBox("Preserve Aspect Ratio")
        self.preserve_aspect_ratio.setChecked(True)
        self.preserve_aspect_ratio.setToolTip("Maintain image aspect ratio when resizing (adds padding)")
        aspect_crop_layout.addWidget(self.preserve_aspect_ratio)
        
        # Random cropping checkbox
        self.random_crop_check = QCheckBox("Random Cropping")
        self.random_crop_check.setChecked(False)
        self.random_crop_check.setToolTip("Generate additional training samples by randomly cropping images")
        self.random_crop_check.stateChanged.connect(self.toggle_random_crop)
        aspect_crop_layout.addWidget(self.random_crop_check)
        
        settings_layout.addRow(aspect_crop_layout)
        
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
    
    def toggle_augmentation(self, state):
        """Enable/disable augmentation settings button based on checkbox state"""
        self.btn_aug_settings.setEnabled(state == Qt.CheckState.Checked.value)
        # Log the action
        if state == Qt.CheckState.Checked.value:
            self.log_output.append("Data augmentation enabled")
        else:
            self.log_output.append("Data augmentation disabled")
        
    def toggle_random_crop(self, state):
        """Disable aspect ratio preservation when random crop is enabled"""
        is_checked = (state == Qt.CheckState.Checked.value)
        self.preserve_aspect_ratio.setEnabled(not is_checked)
        if is_checked:
            self.preserve_aspect_ratio.setChecked(False)
            self.log_output.append("Random cropping enabled - aspect ratio preservation disabled")
        else:
            self.log_output.append("Random cropping disabled - aspect ratio preservation enabled")
            
        # Update the augmentation settings
        self.augmentation_settings["random_crop"] = is_checked
    
    def open_augmentation_settings(self):
        """Open dialog for configuring data augmentation settings"""
        dialog = AugmentationSettingsDialog(self, self.augmentation_settings)
        if dialog.exec():
            old_settings = self.augmentation_settings.copy()
            self.augmentation_settings = dialog.get_settings()
            
            # Update UI if random crop setting changed
            if self.augmentation_settings["random_crop"] != self.random_crop_check.isChecked():
                self.random_crop_check.setChecked(self.augmentation_settings["random_crop"])
            
            # Log the changes
            self.log_output.append("Data augmentation settings updated:")
            for key, value in self.augmentation_settings.items():
                if old_settings.get(key) != value:
                    self.log_output.append(f"  - {key}: {value}")

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
        
        # Clear existing files in train directories
        for file in os.listdir(images_dir):
            os.remove(os.path.join(images_dir, file))
        for file in os.listdir(labels_dir):
            os.remove(os.path.join(labels_dir, file))
        self.log_output.append("Cleared existing training files.")
        
        try:
            # Get confidence threshold (convert from percentage to decimal)
            threshold = self.threshold_input.value() / 100.0
            self.log_output.append(f"Using confidence threshold: {threshold:.2f}")
            
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
            skipped_below_threshold = 0
            
            # Check if we have FSL prediction results
            has_predictions = False
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            fsl_results = {}
            
            # First try loading from memory
            if hasattr(self.parent, 'fsl_tab') and hasattr(self.parent.fsl_tab, 'fsl_results'):
                fsl_results = self.parent.fsl_tab.fsl_results
                has_predictions = bool(fsl_results)
                if has_predictions:
                    self.log_output.append("Found FSL prediction results in memory.")
            
            # If not in memory, try loading from files
            if not has_predictions and os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
                # Load predictions from the results directory
                for filename in os.listdir(results_dir):
                    if filename.endswith(".json"):
                        self.log_output.append(f"Loading predictions from: {filename}")
                        try:
                            with open(os.path.join(results_dir, filename), 'r') as f:
                                data = json.load(f)
                                fsl_results.update(data)
                        except json.JSONDecodeError:
                            self.log_output.append(f"Warning: Could not parse {filename} as JSON")
                            continue
                
                has_predictions = bool(fsl_results)
                if has_predictions:
                    self.log_output.append(f"Loaded FSL prediction results from files. Total frames with predictions: {len(fsl_results)}")
            
            # Get all annotation files
            annotation_files = [f for f in os.listdir(annotations_dir) 
                               if f.startswith("frame_") and f.endswith(".txt")]
            
            # Create and configure progress dialog
            progress_dialog = QProgressDialog("Preparing training data...", "Cancel", 0, len(annotation_files), self)
            progress_dialog.setWindowTitle("Data Preparation")
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setMinimumDuration(0)
            progress_dialog.setValue(0)
            
            # Check class mapping - map class names to indices
            class_name_to_idx = {}
            if hasattr(self.parent, 'classes'):
                class_name_to_idx = {class_name.lower(): i for i, class_name in enumerate(self.parent.classes)}
                
            # Map class names from predictions file to our class indices
            # This handles if prediction file has different class names than our current project
            class_map = {
                'princess': 0,      # Default mappings - update as needed
                'goblin_barrel': 1,
                'furnace': 2
            }
            
            # Override with actual class indices if available
            for class_name, idx in class_name_to_idx.items():
                for pred_class in class_map.keys():
                    if pred_class.lower() == class_name.lower():
                        class_map[pred_class] = idx
            
            # Process each frame with annotations
            for i, annotation_file in enumerate(annotation_files):
                # Update progress dialog
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    self.log_output.append("Data preparation canceled by user.")
                    cap.release()
                    return False
                
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
                
                # Create list to collect valid annotations for this frame
                valid_annotations = []
                
                # Create YOLO format annotation content to be written later if we have valid annotations
                yolo_annotations = []
                
                # Process all annotations in this frame
                for ann in annotations:
                    class_id = ann['class']
                    
                    # Handle different types of annotations
                    if class_id != '-':  # User annotated class
                        # Use the class index directly
                        class_idx = int(class_id) if class_id.isdigit() else 0
                        yolo_annotations.append(f"{class_idx} {ann['bbox_yolo'][0]} {ann['bbox_yolo'][1]} {ann['bbox_yolo'][2]} {ann['bbox_yolo'][3]}")
                        valid_annotations.append(ann)
                        user_annotations += 1
                    elif has_predictions:  # Check if we have predictions for this detection
                        # This is a detection without user annotation, check FSL predictions
                        annotation_id = str(ann['annotation_id'])
                        
                        # Try both with padding and without padding for frame number
                        frame_keys_to_try = [
                            formatted_frame_number,        # "000123"
                            str(frame_idx)                 # "123"
                        ]
                        
                        found_prediction = False
                        best_class_idx = None
                        best_confidence = 0
                        
                        # Try all possible frame keys
                        for frame_key in frame_keys_to_try:
                            if frame_key in fsl_results:
                                # Correct structure navigation:
                                # fsl_results[frame_key][class_name][annotation_id] = confidence
                                
                                # Look through all class predictions for this frame
                                for class_name, annotations_dict in fsl_results[frame_key].items():
                                    # Try to find the annotation_id in this class's annotations
                                    if annotation_id in annotations_dict:
                                        # Get the confidence value
                                        confidence = float(annotations_dict[annotation_id])
                                        
                                        # Keep track of the best class prediction
                                        if confidence > best_confidence:
                                            best_confidence = confidence
                                            # Map the class name to our class index
                                            if class_name in class_map:
                                                best_class_idx = class_map[class_name]
                                            else:
                                                # Default to first class if mapping not found
                                                best_class_idx = 0
                                            found_prediction = True
                        
                        # If we found a prediction and it's above threshold, add it to the valid annotations
                        if found_prediction and best_confidence >= threshold and best_class_idx is not None:
                            yolo_annotations.append(f"{best_class_idx} {ann['bbox_yolo'][0]} {ann['bbox_yolo'][1]} {ann['bbox_yolo'][2]} {ann['bbox_yolo'][3]}")
                            valid_annotations.append(ann)
                            prediction_annotations += 1
                        elif found_prediction:
                            skipped_below_threshold += 1
                
                # Only save the frame if we have at least one valid annotation
                if valid_annotations:
                    # Save the image to YOLO images directory
                    img_path = os.path.join(images_dir, f"{video_name}_{formatted_frame_number}.jpg")
                    cv2.imwrite(img_path, frame)
                    
                    # Create YOLO format annotation file in labels directory
                    label_path = os.path.join(labels_dir, f"{video_name}_{formatted_frame_number}.txt")
                    with open(label_path, 'w') as f:
                        for annotation in yolo_annotations:
                            f.write(f"{annotation}\n")
                    
                    total_frames += 1
                
                total_annotations = user_annotations + prediction_annotations
            
            # Close the progress dialog
            progress_dialog.setValue(len(annotation_files))
            
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
            
            # Generate additional training data with random cropping if enabled
            if self.random_crop_check.isChecked() or self.augmentation_settings.get("random_crop", False):
                self.log_output.append("Generating additional training data with random cropping...")
                self._generate_random_crops(images_dir, labels_dir)
            
            # Show statistics
            self.log_output.append(f"Training data preparation complete:")
            self.log_output.append(f"- Total frames: {total_frames}")
            self.log_output.append(f"- User annotations: {user_annotations}")
            self.log_output.append(f"- Predictions above threshold: {prediction_annotations}")
            if skipped_below_threshold > 0:
                self.log_output.append(f"- Predictions below threshold (skipped): {skipped_below_threshold}")
            self.log_output.append(f"- Total annotations included: {user_annotations + prediction_annotations}")
            self.log_output.append(f"- Dataset saved to: {base_dir}")
            
            # Show dialog with statistics
            QMessageBox.information(self, "Training Data Ready", 
                f"Training data prepared successfully!\n\n"
                f"Total frames: {total_frames}\n"
                f"User annotations: {user_annotations}\n"
                f"Predictions above {self.threshold_input.value()}% threshold: {prediction_annotations}\n"
                f"Predictions below threshold (skipped): {skipped_below_threshold}\n"
                f"Total annotations: {user_annotations + prediction_annotations}")
            
            return True
            
        except Exception as e:
            import traceback
            self.log_output.append(f"Error preparing training data: {str(e)}")
            self.log_output.append(traceback.format_exc())
            QMessageBox.critical(self, "Error", f"Failed to prepare training data: {str(e)}")
            return False
    
    def _generate_random_crops(self, images_dir, labels_dir):
        """Generate additional training data using random cropping"""
        try:
            # Get target image size
            target_size = int(self.img_size_input.currentText())
            
            # List all images and labels
            image_files = [f for f in os.listdir(images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            # Create progress dialog
            progress_dialog = QProgressDialog("Generating random crops...", "Cancel", 0, len(image_files), self)
            progress_dialog.setWindowTitle("Random Crop Generation")
            progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
            progress_dialog.setMinimumDuration(0)
            
            crops_added = 0
            self.log_output.append("Generating random crops from existing images...")
            
            for i, img_file in enumerate(image_files):
                # Update progress
                progress_dialog.setValue(i)
                if progress_dialog.wasCanceled():
                    self.log_output.append("Random crop generation canceled by user.")
                    return
                
                # Get corresponding label file
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_file)
                
                # Check if label file exists
                if not os.path.exists(label_path):
                    continue
                
                # Read image
                img_path = os.path.join(images_dir, img_file)
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                img_h, img_w = img.shape[:2]
                
                # Read labels
                with open(label_path, 'r') as f:
                    labels = [line.strip() for line in f if line.strip()]
                
                # Skip if no labels
                if not labels:
                    continue
                
                # Generate 2 random crops per image
                for crop_idx in range(2):
                    # Only crop if the image is larger than the target size
                    if img_w > target_size and img_h > target_size:
                        # Generate random crop coordinates
                        x1 = max(0, min(img_w - target_size, np.random.randint(0, img_w - target_size)))
                        y1 = max(0, min(img_h - target_size, np.random.randint(0, img_h - target_size)))
                        x2 = x1 + target_size
                        y2 = y1 + target_size
                        
                        # Crop image
                        cropped_img = img[y1:y2, x1:x2]
                        
                        # Create new filenames for the cropped image and labels
                        crop_img_file = f"{os.path.splitext(img_file)[0]}_crop{crop_idx}.jpg"
                        crop_label_file = f"{os.path.splitext(label_file)[0]}_crop{crop_idx}.txt"
                        
                        # Save cropped image
                        cv2.imwrite(os.path.join(images_dir, crop_img_file), cropped_img)
                        
                        # Adjust and save labels
                        valid_labels = []
                        for label in labels:
                            parts = label.strip().split()
                            if len(parts) >= 5:
                                class_id = parts[0]
                                x_center, y_center, width, height = map(float, parts[1:5])
                                
                                # Convert YOLO coordinates to absolute
                                abs_x_center = x_center * img_w
                                abs_y_center = y_center * img_h
                                abs_width = width * img_w
                                abs_height = height * img_h
                                
                                # Check if bbox is within crop
                                bbox_x1 = abs_x_center - abs_width/2
                                bbox_y1 = abs_y_center - abs_height/2
                                bbox_x2 = abs_x_center + abs_width/2
                                bbox_y2 = abs_y_center + abs_height/2
                                
                                # Check if box has significant overlap with crop
                                if (bbox_x2 > x1 and bbox_x1 < x2 and bbox_y2 > y1 and bbox_y1 < y2):
                                    # Crop the bbox
                                    crop_bbox_x1 = max(0, bbox_x1 - x1)
                                    crop_bbox_y1 = max(0, bbox_y1 - y1)
                                    crop_bbox_x2 = min(target_size, bbox_x2 - x1)
                                    crop_bbox_y2 = min(target_size, bbox_y2 - y1)
                                    
                                    # Convert back to YOLO format
                                    crop_width = (crop_bbox_x2 - crop_bbox_x1) / target_size
                                    crop_height = (crop_bbox_y2 - crop_bbox_y1) / target_size
                                    crop_x_center = (crop_bbox_x1 + crop_width * target_size / 2) / target_size
                                    crop_y_center = (crop_bbox_y1 + crop_height * target_size / 2) / target_size
                                    
                                    # Only include if the bbox is still valid
                                    if crop_width > 0.01 and crop_height > 0.01:
                                        valid_labels.append(f"{class_id} {crop_x_center} {crop_y_center} {crop_width} {crop_height}")
                        
                        # Only save if we have valid labels
                        if valid_labels:
                            with open(os.path.join(labels_dir, crop_label_file), 'w') as f:
                                for label in valid_labels:
                                    f.write(f"{label}\n")
                            crops_added += 1
            
            # Close progress dialog
            progress_dialog.setValue(len(image_files))
            self.log_output.append(f"Added {crops_added} random crops as additional training data")
            
        except Exception as e:
            self.log_output.append(f"Error generating random crops: {str(e)}")
    
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
        
        # Add augmentation settings
        if self.augmentation_check.isChecked():
            for key, value in self.augmentation_settings.items():
                if key != "random_crop":  # Handle random_crop separately
                    yolo_config[key] = value
        
        try:
            # Setup training output directory
            training_output = os.path.join(self.datasets_dir, "yolo_training", video_name)
            os.makedirs(training_output, exist_ok=True)
            
            # Pass the output directory to config
            yolo_config["output_dir"] = training_output
            
            # Create YOLOTrainer instance
            trainer = YOLOTrainer(yolo_config)
            
            # Clear log and update UI before training
            self.log_output.clear()
            self.log_output.append("Initializing YOLO training...")
            self.log_output.append(f"Model: {yolo_config['model_type']}")
            self.log_output.append(f"Data: {dataset_yaml}")
            self.log_output.append(f"Output: {training_output}")
            self.toggle_training_controls(True)
            
            # Start training in a thread
            self.training_thread = TrainingThread(trainer, os.path.join(self.datasets_dir, "yolo"), training_output, yolo_config)
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
        self.btn_aug_settings.setEnabled(not is_training and self.augmentation_check.isChecked())
        self.preserve_aspect_ratio.setEnabled(not is_training and not self.random_crop_check.isChecked())
        self.random_crop_check.setEnabled(not is_training)
        
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
        model_path = os.path.join(self.datasets_dir, "yolo_training", video_name, "weights", "best.pt")
        
        if not os.path.exists(model_path):
            # Try alternative path
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
            classes_path = os.path.join(self.datasets_dir, "yolo", "config", "classes.txt")
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
        model_path = os.path.join(self.datasets_dir, "yolo_training", video_name, "weights", "best.pt")
        
        # Check alternative path if needed
        if not os.path.exists(model_path):
            model_path = os.path.join(self.datasets_dir, "yolo_training", video_name, "best.pt")
            if not os.path.exists(model_path):
                QMessageBox.warning(self, "Model Not Found", "Trained model not found. Train a model first.")
                return
        
        try:
            # Import YOLO for inference
            from ultralytics import YOLO
            
            # Load the model
            self.log_output.append(f"Loading model from: {model_path}")
            model = YOLO(model_path)
            
            # Run inference on current frame
            current_frame = self.parent.video_player.get_current_frame()
            if current_frame is not None:
                self.log_output.append("Running inference on current frame...")
                
                # Run prediction
                results = model(current_frame)
                
                # Display results
                result_image = results[0].plot()
                self.parent.video_player.set_image(result_image)
                
                # Log detections
                self.log_output.append(f"Detections: {len(results[0].boxes)}")
                for i, box in enumerate(results[0].boxes):
                    cls_id = int(box.cls[0].item())
                    conf = box.conf[0].item()
                    class_name = self.parent.classes[cls_id] if cls_id < len(self.parent.classes) else f"Class {cls_id}"
                    self.log_output.append(f"  {i+1}: {class_name} ({conf:.2f})")
                
                QMessageBox.information(self, "Inference Complete", 
                                     f"Model detected {len(results[0].boxes)} objects.")
            else:
                QMessageBox.warning(self, "No Frame", "No current frame available for inference.")
                
        except ImportError:
            self.log_output.append("Error: Ultralytics package not installed. Cannot run inference.")
            QMessageBox.critical(self, "Missing Dependency", 
                              "Ultralytics package not installed. Please install it with: pip install ultralytics")
        except Exception as e:
            self.log_output.append(f"Error during inference: {str(e)}")
            QMessageBox.critical(self, "Inference Error", f"Error during inference: {str(e)}")
    
    def append_to_log(self, message):
        """Add a message to the log output"""
        self.log_output.append(message)
        # Auto-scroll to the bottom
        self.log_output.ensureCursorVisible()
        
    def get_video_name(self):
        """Get the name of the current video without extension"""
        if not hasattr(self.parent, 'current_video_path') or not self.parent.current_video_path:
            return None
        return os.path.splitext(os.path.basename(self.parent.current_video_path))[0]
