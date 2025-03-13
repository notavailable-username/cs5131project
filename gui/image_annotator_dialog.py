from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QListWidget, 
                           QLabel, QPushButton, QComboBox, QFileDialog,
                           QInputDialog, QMessageBox, QTableWidget, 
                           QTableWidgetItem, QWidget, QSizePolicy, QApplication)
from PyQt6.QtCore import Qt, QPoint
from PyQt6.QtGui import QImage, QPixmap
import cv2
import os
import json

class ImageAnnotatorDialog(QDialog):
    def __init__(self, parent=None, directory=None):
        super().__init__(parent)
        self.setWindowTitle("Image Annotation Tool")
        self.resize(1200, 800)
        
        # Initialize variables
        self.image_dir = ""
        self.image_files = []
        self.current_image_index = -1
        self.current_image = None
        self.current_image_path = ""
        self.classes = ["person", "car", "bicycle", "dog", "unknown"]
        self.current_class = self.classes[0] if self.classes else ""
        
        # Mouse tracking variables
        self.drawing = False
        self.start_point = QPoint()
        self.end_point = QPoint()
        self.bounding_boxes = []  # [(x1, y1, x2, y2, class_name), ...]
        self.current_box = None
        self.selected_box_index = -1  # Track which box is selected
        
        self.setup_ui()
        
        # Load directory if provided
        if directory:
            self.load_directory(directory)
    
    def setup_ui(self):
        main_layout = QHBoxLayout(self)
        
        # Left panel - image selection with scroll view
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        
        # Image list widget with scroll capability
        left_layout.addWidget(QLabel("Images:"))
        self.image_list = QListWidget()
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.image_list.setSizeAdjustPolicy(QListWidget.SizeAdjustPolicy.AdjustToContents)
        self.image_list.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.image_list)
        
        # Make left panel fit exactly to content size
        left_panel.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding)
        left_layout.setContentsMargins(5, 5, 5, 5)  # Compact margins
        
        # Middle panel - image viewer
        middle_panel = QWidget()
        middle_layout = QVBoxLayout(middle_panel)
        
        # Image display label with flexible size - remove fixed width constraint
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumWidth(400)  # Minimum width instead of fixed
        self.image_label.setMinimumHeight(400)  # Minimum height
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image_label.mousePressEvent = self.mouse_press
        self.image_label.mouseMoveEvent = self.mouse_move
        self.image_label.mouseReleaseEvent = self.mouse_release
        
        middle_layout.addWidget(self.image_label, 1)  # Add stretch factor to fill available height
        middle_panel.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Right panel - controls
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Image navigation
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("Previous")
        self.btn_next = QPushButton("Next")
        self.btn_prev.clicked.connect(self.load_prev_image)
        self.btn_next.clicked.connect(self.load_next_image)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        right_layout.addLayout(nav_layout)
        
        # Class selection
        right_layout.addWidget(QLabel("Select Class:"))
        self.class_combo = QComboBox()
        self.class_combo.addItems(self.classes)
        self.class_combo.currentTextChanged.connect(self.update_current_class)
        right_layout.addWidget(self.class_combo)
        
        # Manage classes button
        self.btn_manage_classes = QPushButton("Manage Classes")
        self.btn_manage_classes.clicked.connect(self.manage_classes)
        right_layout.addWidget(self.btn_manage_classes)
        
        # Bounding box table
        right_layout.addWidget(QLabel("Bounding Boxes:"))
        self.box_table = QTableWidget(0, 5)
        self.box_table.setHorizontalHeaderLabels(["Class", "X", "Y", "Width", "Height"])
        self.box_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        # Make table non-editable
        self.box_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        right_layout.addWidget(self.box_table)
        
        # Remove box button
        self.btn_remove_box = QPushButton("Remove Selected Box")
        self.btn_remove_box.clicked.connect(self.remove_selected_box)
        right_layout.addWidget(self.btn_remove_box)
        
        # Load and save buttons
        self.btn_load_dir = QPushButton("Load Image Directory")
        self.btn_save = QPushButton("Save Annotations")
        self.btn_done = QPushButton("Done")
        self.btn_load_dir.clicked.connect(self.load_image_directory)
        self.btn_save.clicked.connect(self.save_annotations)
        self.btn_done.clicked.connect(self.accept)
        
        right_layout.addWidget(self.btn_load_dir)
        right_layout.addWidget(self.btn_save)
        right_layout.addWidget(self.btn_done)
        right_layout.addStretch()
        
        # Add all panels to main layout with appropriate stretch factors
        main_layout.addWidget(left_panel, 0)   # No horizontal stretch for left panel
        main_layout.addWidget(middle_panel, 3)  # Give middle panel horizontal stretch to fit images
        main_layout.addWidget(right_panel, 1)   # Right panel gets all remaining space
        
        self.setLayout(main_layout)
    
    def load_image_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, "Select Image Directory")
        if dir_path:
            self.load_directory(dir_path)
    
    def load_directory(self, directory):
        if not directory:
            return
            
        self.image_dir = directory
        self.image_files = [f for f in os.listdir(directory) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        if not self.image_files:
            QMessageBox.warning(self, "No Images", "No images found in the selected directory.")
            return
        
        # Populate the image list widget
        self.image_list.clear()
        for img_file in self.image_files:
            self.image_list.addItem(img_file)
            
        self.current_image_index = -1
        if self.image_files:
            self.image_list.setCurrentRow(0)
            self.load_image(0)
            
    def on_image_selected(self, item):
        index = self.image_list.row(item)
        if index >= 0 and index < len(self.image_files):
            self.load_image(index)
    
    def load_image(self, index):
        if 0 <= index < len(self.image_files):
            self.current_image_index = index
            self.current_image_path = os.path.join(self.image_dir, self.image_files[index])
            
            # Load the image
            self.current_image = cv2.imread(self.current_image_path)
            if self.current_image is None:
                return
                
            # Convert from BGR to RGB
            self.current_image = cv2.cvtColor(self.current_image, cv2.COLOR_BGR2RGB)
            
            # Load annotations if they exist
            self.bounding_boxes = []
            annotation_file = os.path.splitext(self.current_image_path)[0] + ".json"
            if os.path.exists(annotation_file):
                try:
                    with open(annotation_file, 'r') as f:
                        self.bounding_boxes = json.load(f)
                except Exception:
                    pass
            
            self.update_display()
            self.update_box_table()
            
            self.setWindowTitle(f"Image Annotation Tool - {self.image_files[index]} ({index+1}/{len(self.image_files)})")
            
            # Highlight the current image in the list
            if self.image_list.currentRow() != index:
                self.image_list.setCurrentRow(index)
                
            # Force layout update
            QApplication.processEvents()
    
    def load_next_image(self):
        if self.current_image_index < len(self.image_files) - 1:
            self.load_image(self.current_image_index + 1)
    
    def load_prev_image(self):
        if self.current_image_index > 0:
            self.load_image(self.current_image_index - 1)
    
    def scale_image_to_fit(self, img, max_width, max_height):
        """Scale image to fit within the given dimensions while maintaining aspect ratio"""
        h, w = img.shape[:2]
        
        # Use actual label dimensions instead of hardcoded width
        max_display_width = max_width
        max_display_height = max_height
        
        # Calculate scaling factor
        scale_w = max_display_width / w if w > max_display_width else 1
        scale_h = max_display_height / h if h > max_display_height else 1
        scale = min(scale_w, scale_h)
        
        # If image is already smaller than max dimensions, don't scale up
        if scale >= 1:
            return img
            
        new_width = int(w * scale)
        new_height = int(h * scale)
        
        # Resize image
        return cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    
    def update_display(self):
        if self.current_image is None:
            return
            
        # Create a copy of the image to draw on
        display_image = self.current_image.copy()
        
        # Draw existing bounding boxes
        for i, box in enumerate(self.bounding_boxes):
            x1, y1, x2, y2, class_name = box
            
            # Use pure blue for normal boxes and pure red for selected box
            if i == self.selected_box_index:
                color = (255, 0, 0)  # Pure red in RGB
                thickness = 3  # Thicker for selected box
            else:
                color = (0, 0, 255)  # Pure blue in RGB
                thickness = 2
            
            cv2.rectangle(display_image, (x1, y1), (x2, y2), color, thickness)
            # Increased text size from 0.5 to 0.8, increased thickness from 1 to 2
            cv2.putText(display_image, class_name, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        
        # Draw the current box being created - also in pure red
        if self.drawing and self.current_box:
            x1, y1, x2, y2 = self.current_box
            cv2.rectangle(display_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Pure red for box being drawn
        
        # Scale image to fit in label
        max_width = self.image_label.width()
        max_height = self.image_label.height()
        display_image = self.scale_image_to_fit(display_image, max_width, max_height)
        
        # Convert to QImage and display
        h, w, c = display_image.shape
        bytes_per_line = c * w
        qimg = QImage(display_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(qimg))
    
    def update_box_table(self):
        self.box_table.setRowCount(0)
        
        if self.current_image is None:
            return  # Don't update if no image is loaded
            
        image_height, image_width = self.current_image.shape[:2]
        
        for i, (x1, y1, x2, y2, class_name) in enumerate(self.bounding_boxes):
            # Convert to YOLO format (normalized)
            x_center = (x1 + x2) / 2 / image_width
            y_center = (y1 + y2) / 2 / image_height
            width = (x2 - x1) / image_width
            height = (y2 - y1) / image_height
            
            row = self.box_table.rowCount()
            self.box_table.insertRow(row)
            
            self.box_table.setItem(row, 0, QTableWidgetItem(class_name))
            self.box_table.setItem(row, 1, QTableWidgetItem(f"{x_center:.4f}"))
            self.box_table.setItem(row, 2, QTableWidgetItem(f"{y_center:.4f}"))
            self.box_table.setItem(row, 3, QTableWidgetItem(f"{width:.4f}"))
            self.box_table.setItem(row, 4, QTableWidgetItem(f"{height:.4f}"))
        
        # Connect row selection signal
        self.box_table.itemSelectionChanged.connect(self.on_box_selection_changed)

    def on_box_selection_changed(self):
        selected_rows = self.box_table.selectionModel().selectedRows()
        if selected_rows:
            self.selected_box_index = selected_rows[0].row()
        else:
            self.selected_box_index = -1
        self.update_display()
    
    def select_box(self, row):
        # Allow editing of selected box
        if 0 <= row < len(self.bounding_boxes):
            self.selected_box_index = row
            # Select the row in the table
            self.box_table.selectRow(row)
            self.update_display()
    
    def remove_selected_box(self):
        selected_rows = set()
        for item in self.box_table.selectedItems():
            selected_rows.add(item.row())
        
        # Remove rows in reverse order to avoid index shifting
        for row in sorted(selected_rows, reverse=True):
            if 0 <= row < len(self.bounding_boxes):
                self.bounding_boxes.pop(row)
        
        self.update_display()
        self.update_box_table()
    
    def update_current_class(self, class_name):
        self.current_class = class_name
    
    def manage_classes(self):
        # Simple class management
        text, ok = QInputDialog.getText(
            self, 'Manage Classes', 
            'Enter class names separated by commas:',
            text=",".join(self.classes)
        )
        
        if ok and text.strip():
            self.classes = [c.strip() for c in text.split(",") if c.strip()]
            self.class_combo.clear()
            self.class_combo.addItems(self.classes)
            if self.classes:
                self.current_class = self.classes[0]
    
    def save_annotations(self):
        if not self.current_image_path:
            QMessageBox.warning(self, "Warning", "No image loaded.")
            return
            
        # No check for empty bounding_boxes - we allow saving even when all boxes are deleted
        
        try:
            # Save annotations in JSON format
            annotation_file = os.path.splitext(self.current_image_path)[0] + ".json"
            os.makedirs(os.path.dirname(annotation_file), exist_ok=True)
            
            with open(annotation_file, 'w') as f:
                json.dump(self.bounding_boxes, f)
                
            # Also save in YOLO format
            yolo_file = os.path.splitext(self.current_image_path)[0] + ".txt"
            os.makedirs(os.path.dirname(yolo_file), exist_ok=True)
            
            # Check if we have a valid image
            if self.current_image is None:
                QMessageBox.warning(self, "Error", "Image data is missing.")
                return
                
            # Get image dimensions for YOLO normalization
            image_height, image_width = self.current_image.shape[:2]
            if image_height <= 0 or image_width <= 0:
                QMessageBox.warning(self, "Error", "Invalid image dimensions.")
                return
                
            with open(yolo_file, 'w') as f:
                for x1, y1, x2, y2, class_name in self.bounding_boxes:
                    # Get class index with better error handling
                    try:
                        class_idx = self.classes.index(class_name)
                    except ValueError:
                        # If class not found, use first class or create an "unknown" class
                        if "unknown" in self.classes:
                            class_idx = self.classes.index("unknown")
                        elif self.classes:
                            class_idx = 0  # Default to first class if not found
                        else:
                            # No classes defined, create a default
                            self.classes = ["unknown"]
                            self.class_combo.clear()
                            self.class_combo.addItems(self.classes)
                            class_idx = 0
                    
                    # Convert to YOLO format (ensure proper normalization)
                    # Make sure coordinates are within image bounds
                    x1 = max(0, min(image_width-1, x1))
                    y1 = max(0, min(image_height-1, y1))
                    x2 = max(0, min(image_width-1, x2))
                    y2 = max(0, min(image_height-1, y2))
                    
                    # Calculate normalized values
                    x_center = (x1 + x2) / 2 / image_width
                    y_center = (y1 + y2) / 2 / image_height
                    width = abs(x2 - x1) / image_width
                    height = abs(y2 - y1) / image_height
                    
                    # Ensure values are within [0,1] range
                    x_center = max(0, min(1, x_center))
                    y_center = max(0, min(1, y_center))
                    width = max(0, min(1, width))
                    height = max(0, min(1, height))
                    
                    # Write to file
                    f.write(f"{class_idx} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
                
            QMessageBox.information(self, "Success", f"Saved annotations to {annotation_file} and {yolo_file}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Error saving annotations: {str(e)}")
            import traceback
            traceback.print_exc()  # Print the full stack trace for debugging
            
    def get_adjusted_mouse_position(self, position):
        """
        Adjust mouse position to account for image scaling within the label
        """
        if self.current_image is None:
            return position
            
        pixmap = self.image_label.pixmap()
        if pixmap is None:
            return position
            
        # Calculate scaling factors
        image_height, image_width = self.current_image.shape[:2]
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        
        # Calculate the position of the image within the label
        label_width = self.image_label.width()
        label_height = self.image_label.height()
        x_offset = (label_width - pixmap_width) / 2
        y_offset = (label_height - pixmap_height) / 2
        
        # Adjust mouse position
        mouse_x = position.x() - x_offset
        mouse_y = position.y() - y_offset
        
        # Scale back to original image coordinates
        if pixmap_width > 0 and pixmap_height > 0:
            scale_x = image_width / pixmap_width
            scale_y = image_height / pixmap_height
            mouse_x = max(0, min(image_width, mouse_x * scale_x))
            mouse_y = max(0, min(image_height, mouse_y * scale_y))
            
        return QPoint(int(mouse_x), int(mouse_y))

    def mouse_press(self, event):
        """Handle mouse press event for starting a bounding box or selecting/deselecting boxes"""
        if self.current_image is None:
            return
            
        # Get position adjusted for any scaling
        pos = self.get_adjusted_mouse_position(event.position())
        
        # Check if we clicked inside an existing box
        clicked_box_index = -1
        for i, (x1, y1, x2, y2, _) in enumerate(self.bounding_boxes):
            if x1 <= pos.x() <= x2 and y1 <= pos.y() <= y2:
                clicked_box_index = i
                break
        
        # If clicked on a box, select it
        if clicked_box_index >= 0:
            self.select_box(clicked_box_index)
            return
        else:
            # If clicked on empty area, deselect current box
            self.selected_box_index = -1
            self.box_table.clearSelection()
        
        # Start drawing a new box
        self.drawing = True
        self.start_point = pos
        self.end_point = pos
        self.current_box = (pos.x(), pos.y(), pos.x(), pos.y())
        self.update_display()
    
    def mouse_move(self, event):
        """Handle mouse move event for updating the bounding box during drawing"""
        if not self.drawing or self.current_image is None:
            return
            
        # Get position adjusted for any scaling
        pos = self.get_adjusted_mouse_position(event.position())
        
        self.end_point = pos
        self.current_box = (self.start_point.x(), self.start_point.y(), pos.x(), pos.y())
        self.update_display()
    
    def mouse_release(self, event):
        """Handle mouse release event for finalizing the bounding box"""
        if not self.drawing or self.current_image is None:
            return
            
        # Get position adjusted for any scaling
        pos = self.get_adjusted_mouse_position(event.position())
        
        # Make sure end point is updated
        self.end_point = pos
        self.drawing = False
        
        # Create box with properly ordered coordinates (ensure x1 < x2 and y1 < y2)
        x1 = min(self.start_point.x(), self.end_point.x())
        y1 = min(self.start_point.y(), self.end_point.y())
        x2 = max(self.start_point.x(), self.end_point.x())
        y2 = max(self.start_point.y(), self.end_point.y())
        
        # Add box if it has a minimum size
        if x2 - x1 > 5 and y2 - y1 > 5:
            # Add the current class name to the box
            self.bounding_boxes.append((x1, y1, x2, y2, self.current_class))
            self.update_display()
            self.update_box_table()
        
        self.current_box = None
