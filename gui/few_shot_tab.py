import os
import json
import cv2
import shutil  # Add this import for directory operations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QComboBox, QMessageBox, QProgressBar, QSizePolicy,
    QSpinBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread

class FewShotTab(QWidget):
    """Widget for few-shot learning tab in the main window"""
    
    annotation_saved = pyqtSignal(int, list)  # Frame index, annotations
    request_frame_seek = pyqtSignal(int)  # Frame index
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.current_annotations = []
        self.class_map = {}  # Map class IDs to class names
        self.setupUI()
    
    def setupUI(self):
        layout = QVBoxLayout()
        
        # Class selector
        layout.addWidget(QLabel("Select class to assign:"))
        self.class_selector = QComboBox()
        layout.addWidget(self.class_selector)
        
        # Navigation buttons
        nav_layout = QHBoxLayout()
        
        self.btn_prev_annotated = QPushButton("Previous Annotated Frame")
        self.btn_prev_annotated.clicked.connect(self.goto_prev_annotated_frame)
        nav_layout.addWidget(self.btn_prev_annotated)
        
        self.btn_next_annotated = QPushButton("Next Annotated Frame")
        self.btn_next_annotated.clicked.connect(self.goto_next_annotated_frame)
        nav_layout.addWidget(self.btn_next_annotated)
        
        layout.addLayout(nav_layout)
        
        # Create table for Select/Class columns
        layout.addWidget(QLabel("Bounding Box Annotations:"))
        self.annotation_table = QTableWidget(0, 2)
        self.annotation_table.setHorizontalHeaderLabels(["Select", "Class"])
        self.annotation_table.setColumnWidth(0, 60)
        self.annotation_table.setColumnWidth(1, 150)
        self.annotation_table.horizontalHeader().setStretchLastSection(True)
        self.annotation_table.verticalHeader().setVisible(False)
        self.annotation_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        
        self.annotation_table.cellClicked.connect(self.on_annotation_cell_clicked)
        layout.addWidget(self.annotation_table)

        export = QHBoxLayout()
        
        # Add Export Support Images button
        self.btn_export_supports = QPushButton("Export Support and Query Images")
        self.btn_export_supports.clicked.connect(self.export_support_and_query_images)
        export.addWidget(self.btn_export_supports)

        self.export_progress = QProgressBar()
        self.export_progress.setMinimumWidth(250)
        self.export_progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.export_progress.setValue(0)
        export.addWidget(self.export_progress)
        layout.addLayout(export)

        self.btn_train = QPushButton("Train")
        self.btn_train.clicked.connect(self.train_fsl)
        layout.addWidget(self.btn_train)

        self.train_progress = QProgressBar()
        self.train_progress.setMinimumWidth(250)
        self.train_progress.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.train_progress.setValue(0)
        layout.addWidget(self.train_progress)

        threshold = QHBoxLayout()

        self.label = QLabel("Threshold:")
        threshold.addWidget(self.label)

        self.threshold = QSpinBox()
        self.threshold.setRange(0, 100)
        self.threshold.setSuffix("%")
        self.threshold.setValue(50)  # Default value of 50%
        self.threshold.valueChanged.connect(self.on_threshold_changed)  # Connect value changed signal
        threshold.addWidget(self.threshold)
        layout.addLayout(threshold)

        self.btn_toggle_view = QPushButton("Show Predictions")
        self.btn_toggle_view.setCheckable(True)
        self.btn_toggle_view.clicked.connect(self.toggle_prediction_view)
        layout.addWidget(self.btn_toggle_view)
        
        self.setLayout(layout)

    def on_threshold_changed(self, value):
        """Handle threshold value changes"""
        results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
        # Only update view if currently showing predictions
        if self.btn_toggle_view.isChecked():
            current_frame_idx = self.main_window.video_player.get_current_frame_idx()
            self.load_and_display_annotations(current_frame_idx, show_predictions=True)

    def toggle_prediction_view(self):
        """Toggle between showing annotations and predictions"""
        current_frame = self.main_window.video_player.get_current_frame_idx()
        print(f"Current frame: {current_frame}")
        if self.btn_toggle_view.isChecked():
            self.btn_toggle_view.setText("Show Annotations")
            self.load_and_display_annotations(current_frame, show_predictions=True)
        else:
            self.btn_toggle_view.setText("Show Predictions")
            # Show original annotations
            self.load_and_display_annotations(current_frame)

    def load_and_display_annotations(self, frame_idx, show_predictions=False):
        """
        Unified function to load and display frame data (annotations and predictions if available).

        Args:
            frame_idx: The frame index to load data for
            show_predictions: If True, overlay prediction data on annotations
        """
        if not self.main_window or not self.main_window.current_video_path:
            return

        # Set the appropriate view type in the video player
        view_type = "Prediction View" if show_predictions else "Annotation View"
        self.main_window.video_player.set_video_source_type(view_type)

        # Get and validate the annotations directory
        annotations_dir = self.main_window.get_annotations_dir_for_current_video()
        if not annotations_dir:
            return

        # Construct the annotation file path
        annotation_file = os.path.join(annotations_dir, f"frame_{frame_idx:06d}.txt")
        if not os.path.exists(annotation_file):
            self.main_window.video_player.clear_annotations()
            self.update_annotation_table([])
            self.current_annotations = []
            return

        # Get current frame dimensions
        current_frame = self.main_window.video_player.get_current_frame()
        if current_frame is None:
            return

        frame_h, frame_w = current_frame.shape[:2]

        # Load annotations
        annotations = []
        with open(annotation_file, 'r') as f:
            counter = 0
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    try:
                        class_id = parts[0]
                        x_center, y_center, width, height = map(float, parts[1:5])
                        x1 = int((x_center - width / 2) * frame_w)
                        y1 = int((y_center - height / 2) * frame_h)
                        x2 = int((x_center + width / 2) * frame_w)
                        y2 = int((y_center + height / 2) * frame_h)
                        
                        # Create annotation entry with is_user_annotated flag
                        annotation = {
                            'bbox_abs': [max(0, x1), max(0, y1), min(frame_w - 1, x2), min(frame_h - 1, y2)],
                            'class': class_id,
                            'bbox_yolo': [x_center, y_center, width, height],
                            'annotation_id': counter,
                            'is_user_annotated': class_id != '-'  # Flag to identify user-annotated boxes
                        }
                        
                        # Add class_name for display (translate from ID using class_map)
                        if class_id != '-' and class_id in self.class_map:
                            annotation['class_name'] = self.class_map.get(class_id, class_id)
                        
                        annotations.append(annotation)
                        counter += 1
                    except ValueError:
                        continue

        # Update current annotations
        self.current_annotations = annotations

        # If showing predictions, add prediction data to annotations
        if show_predictions:
            # Get prediction results
            results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
            results = {}

            # Check if we have stored results from a previous training session
            if hasattr(self, 'fsl_results'):
                results = self.fsl_results
            # Otherwise try to load from files
            elif os.path.exists(results_dir) and len(os.listdir(results_dir)) > 0:
                for filename in os.listdir(results_dir):
                    if filename.endswith(".json"):
                        with open(os.path.join(results_dir, filename), 'r') as f:
                            data = json.load(f)
                            results.update(data)

            # Get prediction data for this frame
            frame_key = f"{frame_idx:06d}"
            frame_predictions = results.get(frame_key, {})

            # Get threshold value
            threshold = self.threshold.value() / 100.0

            # Add prediction info to annotations
            for ann in annotations:
                ann_id = str(ann.get('annotation_id', ''))

                # Find the best prediction for this annotation
                best_class = None
                best_confidence = 0.0
                all_predictions = {}

                for class_name, class_predictions in frame_predictions.items():
                    if ann_id in class_predictions:
                        confidence = class_predictions[ann_id]
                        all_predictions[class_name] = confidence
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_class = class_name

                # Add prediction data to annotation
                if best_class and best_confidence >= threshold:
                    ann["is_predicted"] = True
                    ann["predicted_class"] = best_class
                    ann["confidence"] = best_confidence
                    ann["all_predictions"] = all_predictions
                else:
                    ann["is_predicted"] = False

        # Display annotations in the video player
        self.main_window.video_player.annotate_current_frame(annotations, show_prediction=show_predictions)

        # Update the table (different method based on mode)
        if show_predictions:
            self.update_prediction_table(annotations)
        else:
            self.update_annotation_table(annotations)

    def update_annotation_table(self, annotations):
        self.annotation_table.setRowCount(0)
        for i, annotation in enumerate(annotations):
            self.annotation_table.insertRow(i)
            
            # Always show "Select" in the select column for annotation mode
            select_item = QTableWidgetItem("Select")
            select_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.annotation_table.setItem(i, 0, select_item)
            
            # Display class name instead of ID if available
            class_id = annotation.get('class', '-')
            class_name = self.class_map.get(class_id, class_id)  # Use ID as fallback if no name found
            
            class_item = QTableWidgetItem(class_name)
            class_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.annotation_table.setItem(i, 1, class_item)
    
    def update_prediction_table(self, predictions):
        """Update the table to show prediction results."""
        self.annotation_table.setRowCount(0)

        for i, prediction in enumerate(predictions):
            self.annotation_table.insertRow(i)
            
            # Column 0: Select - Show different values based on annotation type
            if prediction.get('is_user_annotated', False):
                # User annotated boxes
                select_item = QTableWidgetItem("true")
            elif prediction.get('is_predicted', False) and prediction.get('confidence', 0) >= self.threshold.value() / 100.0:
                # Predicted boxes above threshold - show confidence
                confidence_percent = prediction.get('confidence', 0) * 100
                select_item = QTableWidgetItem(f"{confidence_percent:.1f}%")
            else:
                # Boxes below threshold or not predicted
                select_item = QTableWidgetItem("-")
                
            select_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.annotation_table.setItem(i, 0, select_item)

            # Column 1: Class - Show appropriate class based on annotation type
            if prediction.get('is_user_annotated', False):
                # For user annotations, show class (true)
                class_id = prediction.get('class', '-')
                class_name = self.class_map.get(class_id, class_id)
                display_text = f"{class_name} (true)"
            elif prediction.get('is_predicted', False) and prediction.get('confidence', 0) >= self.threshold.value() / 100.0:
                # For predictions above threshold, show predicted class
                pred_class = prediction.get('predicted_class', '-')
                class_name = self.class_map.get(pred_class, pred_class)
                display_text = f"{class_name} (predicted)"
            else:
                # For boxes below threshold
                display_text = "-"

            class_item = QTableWidgetItem(display_text)

            # Create tooltip with all predictions if available
            if prediction.get('all_predictions'):
                all_preds = prediction.get('all_predictions', {})
                tooltip_lines = []
                for pred_class, conf in sorted(all_preds.items(), key=lambda x: x[1], reverse=True):
                    pred_name = self.class_map.get(pred_class, pred_class)
                    tooltip_lines.append(f"{pred_name}: {conf*100:.1f}%")
                class_item.setToolTip("\n".join(tooltip_lines))

            class_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.annotation_table.setItem(i, 1, class_item)

    def train_fsl(self):
        """
        Train the few-shot learning model using exported support and query images.
        Uses the FewShotPredictor class from inference.py to make predictions.
        """
        try:
            # Check if support and query images have been exported
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
            support_base_dir = os.path.join(base_dir, "supports")
            query_dir = os.path.join(base_dir, "queries")
            
            if not os.path.exists(support_base_dir) or not os.path.exists(query_dir):
                QMessageBox.warning(self, "No Data", 
                                   "Please export support and query images first by clicking the 'Export Support and Query Images' button.")
                return
                
            # Check if there are any support images
            support_classes = [d for d in os.listdir(support_base_dir) if os.path.isdir(os.path.join(support_base_dir, d))]
            if not support_classes:
                QMessageBox.warning(self, "No Support Images", 
                                   "No support image classes found. Please ensure you have exported support images.")
                return
                
            # Check if there are any query images
            query_images = [f for f in os.listdir(query_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            if not query_images:
                QMessageBox.warning(self, "No Query Images", 
                                   "No query images found. Please ensure you have exported query images.")
                return
            
            # Disable the train button and update text
            self.btn_train.setEnabled(False)
            self.btn_train.setText("Training in Progress...")
            
            # Reset progress bar
            self.train_progress.setValue(0)
            
            # Create and start the training thread
            self.train_thread = TrainFSLThread(
                support_base_dir,
                query_dir,
                self.threshold.value() / 100.0  # Convert percent to decimal
            )
            
            # Connect signals
            self.train_thread.trainProgress.connect(self.update_train_progress)
            self.train_thread.trainComplete.connect(self.handle_train_complete)
            self.train_thread.trainError.connect(self.handle_train_error)
            
            # Start the thread
            self.train_thread.start()
            
        except Exception as e:
            QMessageBox.critical(self, "Training Error", f"An error occurred: {str(e)}")
            self.btn_train.setEnabled(True)
            self.btn_train.setText("Train")

    def update_train_progress(self, value):
        """Update the progress bar during training."""
        self.train_progress.setValue(value)

    def handle_train_complete(self, results_path):
        """Handle completion of training and results display."""
        self.btn_train.setEnabled(True)
        self.btn_train.setText("Train")

        try:
            #Load the training results from the json file
            with open(results_path, 'r') as f:
                results = json.load(f)
            
            self.fsl_results = self._organize_results_by_frame(results)

            #Display predictions for the current frame
            current_frame_idx = self.main_window.video_player.get_current_frame_idx()
            self.load_and_display_predictions(current_frame_idx, self.fsl_results)

            QMessageBox.information(
                self,
                "Training Complete",
                f"FSL training completed successfully!\nResults saved to: {results_path}"
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error Loading Results",
                f"Could not load results from {results_path}: {str(e)}"
            )

    def _organize_results_by_frame(self, results):
        """Organize the flat results list into a nested dictionary by frame and annotation index."""
        organized = {}

        for filename, prediction in results.items():
            # Extract frame number and annotation index from filename
            # Format is: frame_{frame_num}_{annotation_idx}.png
            parts = filename.split('_')

            if len(parts) >= 3:
                frame_num = parts[1]
                annotation_idx = parts[2].split('.')[0]  # Remove extension

                if frame_num not in organized:
                    organized[frame_num] = {}

                organized[frame_num][annotation_idx] = prediction['all_predictions']

        return organized
    
    def handle_train_error(self, error_message):
        """Handle errors during training."""
        self.btn_train.setEnabled(True)
        self.btn_train.setText("Train")
        QMessageBox.critical(self, "Training Error", error_message)
    
    def update_classes(self, classes):
        self.class_selector.clear()
        self.class_selector.addItem('-')  # Add placeholder option for removing class
        self.class_selector.addItems(classes)
        
        # Update class mapping
        self.class_map = {str(i): class_name for i, class_name in enumerate(classes)}
        self.class_map['-'] = '-'  # Add mapping for placeholder
    
    def extract_frame_numbers(self, annotations_dir):
        if not os.path.exists(annotations_dir):
            return []
        annotation_files = [f for f in os.listdir(annotations_dir) if f.startswith("frame_") and f.endswith(".txt")]
        frame_numbers = []
        for file in annotation_files:
            try:
                frame_num = int(file.replace("frame_", "").replace(".txt", ""))
                frame_numbers.append(frame_num)
            except ValueError:
                continue
        return sorted(frame_numbers)
    
    def goto_prev_annotated_frame(self):
        if not self.main_window or not self.main_window.current_video_path:
            return

        current_frame = self.main_window.video_player.get_current_frame_idx()
        annotations_dir = self.main_window.get_annotations_dir_for_current_video()
        frame_numbers = self.extract_frame_numbers(annotations_dir)

        if not frame_numbers:
            QMessageBox.information(self, "No Annotations", "No annotations found for this video.")
            return

        prev_frame = next((frame for frame in reversed(frame_numbers) if frame < current_frame), None)
        # If no previous frame, loop to the last frame
        if prev_frame is None and frame_numbers:
            prev_frame = frame_numbers[-1]

        if prev_frame is not None:
            # First, seek to the frame
            self.request_frame_seek.emit(prev_frame)

            # Then load and display with the current view mode
            self.load_and_display_annotations(prev_frame, self.btn_toggle_view.isChecked())

    def goto_next_annotated_frame(self):
        if not self.main_window or not self.main_window.current_video_path:
            return

        current_frame = self.main_window.video_player.get_current_frame_idx()
        annotations_dir = self.main_window.get_annotations_dir_for_current_video()
        frame_numbers = self.extract_frame_numbers(annotations_dir)

        if not frame_numbers:
            QMessageBox.information(self, "No Annotations", "No annotations found for this video.")
            return

        next_frame = next((frame for frame in frame_numbers if frame > current_frame), None)
        # If no next frame, loop to the first frame
        if next_frame is None and frame_numbers:
            next_frame = frame_numbers[0]

        if next_frame is not None:
            # First, seek to the frame
            self.request_frame_seek.emit(next_frame)

            # Then load and display with the current view mode
            self.load_and_display_annotations(next_frame, self.btn_toggle_view.isChecked())

    def on_annotation_cell_clicked(self, row, column):
        if not self.current_annotations or row >= len(self.current_annotations):
            return

        current_frame_idx = self.main_window.video_player.get_current_frame_idx()

        # Check if we're in prediction mode or annotation mode
        if self.btn_toggle_view.isChecked():  # Prediction mode
            # Highlight the selected prediction in the video display
            predictions = self.current_annotations.copy()
            for i, pred in enumerate(predictions):
                pred['selected'] = (i == row)

            # Use a different method to highlight predictions or reuse existing with a flag
            self.main_window.video_player.annotate_current_frame(predictions, show_prediction=True)
        else:  # Annotation mode - existing behavior
            if column == 0:
                annotations = self.current_annotations.copy()
                for i, ann in enumerate(annotations):
                    ann['selected'] = (i == row)
                self.main_window.video_player.annotate_current_frame(annotations)
            elif column == 1:
                selected_class = self.class_selector.currentText()
                if not selected_class:
                    QMessageBox.warning(self, "No Class Selected", "Please select a class first.")
                    return
                classes = self.main_window.classes
                class_id = str(classes.index(selected_class)) if selected_class in classes else selected_class
                self.current_annotations[row]['class'] = class_id

                # Update table with class name instead of ID
                self.annotation_table.item(row, 1).setText(selected_class)

                self.main_window.video_player.annotate_current_frame(self.current_annotations)
                self.save_current_annotations(current_frame_idx)
    
    def save_current_annotations(self, frame_idx):
        if not self.main_window.current_video_path or not self.current_annotations:
            return
        annotations_dir = self.main_window.get_annotations_dir_for_current_video()
        if not annotations_dir:
            return
        annotation_file = os.path.join(annotations_dir, f"frame_{frame_idx:06d}.txt")
        with open(annotation_file, 'w') as f:
            for annotation in self.current_annotations:
                class_id = annotation.get('class', '-')
                bbox_yolo = annotation.get('bbox_yolo', [0, 0, 0, 0])
                f.write(f"{class_id} {bbox_yolo[0]:.6f} {bbox_yolo[1]:.6f} {bbox_yolo[2]:.6f} {bbox_yolo[3]:.6f}\n")
        self.annotation_saved.emit(frame_idx, self.current_annotations)
    
    def clear_annotations(self):
        self.current_annotations = []
        self.annotation_table.setRowCount(0)
        if self.main_window and hasattr(self.main_window, "video_player"):
            self.main_window.video_player.clear_annotations()
    
    def export_support_and_query_images(self):
        if not self.main_window or not self.main_window.current_video_path:
            QMessageBox.warning(self, "No Data", "No video or annotations available to export.")
            return

        # Get the directory where all annotation files are stored
        annotations_dir = self.main_window.get_annotations_dir_for_current_video()
        if not annotations_dir or not os.path.exists(annotations_dir):
            QMessageBox.warning(self, "No Annotations", "No annotation directory found.")
            return

        # Reset progress bar
        self.export_progress.setValue(0)

        # Create and configure the export thread
        self.export_thread = ExportThread(
            self.main_window.current_video_path,
            annotations_dir,
            self.class_map
        )

        # Connect signals
        self.export_thread.exportProgress.connect(self.update_export_progress)
        self.export_thread.exportComplete.connect(self.handle_export_complete)
        self.export_thread.exportError.connect(self.handle_export_error)

        # Disable export button while processing
        self.btn_export_supports.setEnabled(False)
        self.btn_export_supports.setText("Exporting...")

        # Start the thread
        self.export_thread.start()

    def update_export_progress(self, value):
        self.export_progress.setValue(value)

    def handle_export_complete(self, support_count, query_count, processed_frames):
        # Re-enable export button
        self.btn_export_supports.setEnabled(True)
        self.btn_export_supports.setText("Export Support and Query Images")

        # Show completion message
        base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
        if support_count > 0 or query_count > 0:
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported {support_count} support images and {query_count} query images "
                f"from {processed_frames} frames to {base_dir}"
            )
        else:
            QMessageBox.information(
                self,
                "No Images Exported",
                "No annotations found to export."
            )

    def handle_export_error(self, error_message):
        # Re-enable export button
        self.btn_export_supports.setEnabled(True)
        self.btn_export_supports.setText("Export Support and Query Images")

        # Show error message
        QMessageBox.critical(self, "Export Error", error_message)

# First, let's modify your ExportThread class to handle the exporting
class ExportThread(QThread):
    exportProgress = pyqtSignal(int)
    exportComplete = pyqtSignal(int, int, int)  # support_count, query_count, processed_frames
    exportError = pyqtSignal(str)
    
    def __init__(self, video_path, annotations_dir, class_map):
        super().__init__()
        self.video_path = video_path
        self.annotations_dir = annotations_dir
        self.class_map = class_map
        self.is_running = True
    
    def stop(self):
        self.is_running = False
    
    def run(self):
        try:
            # Create base directories
            base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
            support_base_dir = os.path.join(base_dir, "supports")
            query_dir = os.path.join(base_dir, "queries")
            
            # Clear previous exports by removing and recreating directories
            if os.path.exists(support_base_dir):
                shutil.rmtree(support_base_dir)
            if os.path.exists(query_dir):
                shutil.rmtree(query_dir)
            
            # Create fresh directories
            os.makedirs(support_base_dir, exist_ok=True)
            os.makedirs(query_dir, exist_ok=True)
            
            # Open the video file
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.exportError.emit("Could not open the video file.")
                return
            
            # Get video dimensions
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Track exported counts
            support_count = 0
            query_count = 0
            processed_frames = 0
            
            # Get all annotation files
            annotation_files = [f for f in os.listdir(self.annotations_dir) if f.startswith("frame_") and f.endswith(".txt")]
            total_files = len(annotation_files)
            
            for i, annotation_file in enumerate(annotation_files):
                if not self.is_running:
                    break
                
                try:
                    # Extract frame number from filename
                    frame_idx = int(annotation_file.replace("frame_", "").replace(".txt", ""))
                    # Format frame number to 6 digits
                    formatted_frame_number = f"{frame_idx:06d}"
                    
                    # Read all annotations with their positions in the file
                    annotations = []
                    with open(os.path.join(self.annotations_dir, annotation_file), 'r') as f:
                        for annotation_idx, line in enumerate(f):
                            parts = line.strip().split()
                            if len(parts) >= 5:
                                try:
                                    class_id = parts[0]
                                    x_center, y_center, width, height = map(float, parts[1:5])
                                    # Convert YOLO coordinates to absolute pixel coordinates
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
                        
                        # Process all annotated bounding boxes in this frame
                        for ann in annotations:
                            class_id = ann.get('class')
                            annotation_idx = ann.get('annotation_idx')
                            formatted_annotation_idx = f"{annotation_idx:04d}"
                            
                            # Get absolute coordinates of the bounding box
                            x1, y1, x2, y2 = ann['bbox_abs']
                            
                            # Crop the image
                            cropped_img = frame[y1:y2, x1:x2]
                            
                            # Create filename with frame number and annotation index
                            filename = f"{formatted_frame_number}_{formatted_annotation_idx}.png"
                            
                            # Handle based on whether it's a support or query image
                            if class_id == '-':  # Query image (placeholder class)
                                filepath = os.path.join(query_dir, filename)
                                cv2.imwrite(filepath, cropped_img)
                                query_count += 1
                            else:  # Support image
                                # Get class name from the class map
                                class_name = self.class_map.get(class_id, f"class_{class_id}")
                                
                                # Create directory for this class if it doesn't exist
                                class_dir = os.path.join(support_base_dir, f"{class_id}_{class_name}")
                                os.makedirs(class_dir, exist_ok=True)
                                
                                # Save the image in the class directory
                                filepath = os.path.join(class_dir, filename)
                                cv2.imwrite(filepath, cropped_img)
                                support_count += 1
                        
                        processed_frames += 1
                    
                    # Update progress
                    progress = int((i + 1) / total_files * 100)
                    self.exportProgress.emit(progress)
                    
                except Exception as e:
                    print(f"Error processing frame {annotation_file}: {str(e)}")
            
            # Release the video capture
            cap.release()
            
            # Signal completion
            self.exportComplete.emit(support_count, query_count, processed_frames)
            
        except Exception as e:
            self.exportError.emit(f"Export error: {str(e)}")

class TrainFSLThread(QThread):
    """Thread for handling the few-shot learning training process."""
    trainProgress = pyqtSignal(int)
    trainComplete = pyqtSignal(str)  # List of prediction results
    trainError = pyqtSignal(str)
    
    def __init__(self, support_dir, query_dir, threshold=0.5):
        super().__init__()
        self.support_dir = support_dir
        self.query_dir = query_dir
        self.threshold = threshold
        self.is_running = True
    
    def stop(self):
        self.is_running = False
    
    def run(self):
        try:
            # Import here to avoid circular imports
            import sys
            import os
            import re
            from models.few_shot_models.inference import inference

            self.trainProgress.emit(20)
            output_path = inference()
            self.trainProgress.emit(100)
            
            # Signal completion
            self.trainComplete.emit(output_path)
            
        except Exception as e:
            print(f"Training error: {str(e)}")
            self.trainError.emit(f"Training error: {str(e)}") 