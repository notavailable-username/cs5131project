import os
import cv2
import shutil  # Add this import for directory operations
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QComboBox, QMessageBox, QProgressBar, QSizePolicy
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
        
        # Add Export Support Images button
        self.btn_export_supports = QPushButton("Export Support and Query Images")
        self.btn_export_supports.clicked.connect(self.export_support_and_query_images)
        layout.addWidget(self.btn_export_supports)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumWidth(250)
        self.progress_bar.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)
        
        self.setLayout(layout)
    
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
        if prev_frame is not None:
            self.request_frame_seek.emit(prev_frame)
            self.load_and_display_annotations(prev_frame)
        elif frame_numbers:
            self.request_frame_seek.emit(frame_numbers[-1])
            self.load_and_display_annotations(frame_numbers[-1])
    
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
        if next_frame is not None:
            self.request_frame_seek.emit(next_frame)
            self.load_and_display_annotations(next_frame)
        elif frame_numbers:
            self.request_frame_seek.emit(frame_numbers[0])
            self.load_and_display_annotations(frame_numbers[0])
    
    def load_and_display_annotations(self, frame_idx):
        if not self.main_window or not self.main_window.current_video_path:
            return
        self.main_window.video_player.set_video_source_type("Annotation View")
        annotations_dir = self.main_window.get_annotations_dir_for_current_video()
        if not annotations_dir:
            return
        annotation_file = os.path.join(annotations_dir, f"frame_{frame_idx:06d}.txt")
        if not os.path.exists(annotation_file):
            self.main_window.video_player.clear_annotations()
            self.update_annotation_table([])
            return
        current_frame = self.main_window.video_player.get_current_frame()
        if current_frame is None:
            return
        frame_h, frame_w = current_frame.shape[:2]
        annotations = []
        with open(annotation_file, 'r') as f:
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
                        annotations.append({
                            'bbox_abs': [max(0, x1), max(0, y1), min(frame_w - 1, x2), min(frame_h - 1, y2)],
                            'class': class_id,
                            'bbox_yolo': [x_center, y_center, width, height]
                        })
                    except ValueError:
                        continue
        self.main_window.video_player.annotate_current_frame(annotations)
        self.update_annotation_table(annotations)
        self.current_annotations = annotations
    
    def update_annotation_table(self, annotations):
        self.annotation_table.setRowCount(0)
        for i, annotation in enumerate(annotations):
            self.annotation_table.insertRow(i)
            select_item = QTableWidgetItem("Select")
            select_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.annotation_table.setItem(i, 0, select_item)
            
            # Display class name instead of ID if available
            class_id = annotation.get('class', '-')
            class_name = self.class_map.get(class_id, class_id)  # Use ID as fallback if no name found
            
            class_item = QTableWidgetItem(class_name)
            class_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            self.annotation_table.setItem(i, 1, class_item)
    
    def on_annotation_cell_clicked(self, row, column):
        if not self.current_annotations or row >= len(self.current_annotations):
            return
        current_frame_idx = self.main_window.video_player.get_current_frame_idx()
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
        self.progress_bar.setValue(0)
        
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
        self.progress_bar.setValue(value)
    
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