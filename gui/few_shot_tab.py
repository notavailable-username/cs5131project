import os
import cv2
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, 
    QTableWidget, QTableWidgetItem, QComboBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal

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
        
        self.setLayout(layout)
    
    def update_classes(self, classes):
        self.class_selector.clear()
        self.class_selector.addItems(classes)
        
        # Update class mapping
        self.class_map = {str(i): class_name for i, class_name in enumerate(classes)}
    
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
