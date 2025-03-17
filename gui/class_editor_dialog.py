from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QListWidget, QTableWidget, 
    QTableWidgetItem, QPushButton, QLabel, QHeaderView, QSplitter,
    QMessageBox, QDialogButtonBox, QWidget, QInputDialog
)
from PyQt6.QtCore import Qt, pyqtSignal
import os
import csv

class ClassEditorDialog(QDialog):
    """Dialog for editing class definitions across different videos"""
    
    classes_updated = pyqtSignal(str, list)  # Signal emitted when classes are updated (video_path, classes)
    
    def __init__(self, parent=None, annotations_dir=None, videos=None):
        super().__init__(parent)
        self.setWindowTitle("Class Editor")
        self.resize(800, 500)
        
        self.annotations_dir = annotations_dir
        self.videos = videos or []
        self.current_video_path = None
        
        # Store classes per video
        self.video_classes = {}  # Dictionary mapping video paths to class lists
        
        self._init_ui()
        self._populate_videos()
    
    def _init_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(5)  # Reduce spacing to save space
        main_layout.setContentsMargins(10, 10, 10, 10)  # Reduce margins
        
        # Title/Header
        header_label = QLabel("Edit Classes for Videos")
        header_label.setStyleSheet("font-size: 16px; font-weight: bold;")
        main_layout.addWidget(header_label)
        
        # Splitter for left and right panels
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter, 1)  # Give it a stretch factor to expand
        
        # Left panel - Video list
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 0, 0)  # Remove padding
        left_layout.setSpacing(3)  # Reduce spacing
        
        video_label = QLabel("Available Videos:")
        left_layout.addWidget(video_label)
        
        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self.on_video_selected)
        left_layout.addWidget(self.video_list, 1)  # Give it a stretch factor
        
        # Right panel - Class table and controls
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)  # Remove padding
        right_layout.setSpacing(3)  # Reduce spacing
        
        classes_label = QLabel("Classes:")
        right_layout.addWidget(classes_label)
        
        self.class_table = QTableWidget(0, 2)
        self.class_table.setHorizontalHeaderLabels(["Index", "Class Name"])
        self.class_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.class_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)  # Select entire rows
        self.class_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)  # Make non-editable
        self.class_table.verticalHeader().setVisible(False)  # Hide default row indices
        right_layout.addWidget(self.class_table, 1)  # Give it a stretch factor
        
        # Buttons for adding/removing classes
        buttons_layout = QHBoxLayout()
        
        self.btn_add_class = QPushButton("Add Class")
        self.btn_add_class.clicked.connect(self.add_class)
        buttons_layout.addWidget(self.btn_add_class)
        
        self.btn_delete_class = QPushButton("Delete Selected Class")
        self.btn_delete_class.clicked.connect(self.delete_class)
        buttons_layout.addWidget(self.btn_delete_class)
        
        right_layout.addLayout(buttons_layout)
        
        # Add containers to splitter
        splitter.addWidget(left_container)
        splitter.addWidget(right_container)
        
        # Set initial splitter sizes
        splitter.setSizes([300, 500])
        
        # Dialog buttons
        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | 
                                     QDialogButtonBox.StandardButton.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        main_layout.addWidget(button_box)
        
        # Apply styling
        self.style_ui_components()
    
    def style_ui_components(self):
        """Apply consistent styling to dialog components"""
        # Button style
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
        
        # Apply to all buttons
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)
    
    def _populate_videos(self):
        """Populate the video list with available videos"""
        self.video_list.clear()
        
        if not self.videos:
            return
            
        for video in self.videos:
            if isinstance(video, dict) and 'path' in video and 'name' in video:
                # Using the video_player's video format
                self.video_list.addItem(video['name'])
            elif isinstance(video, str):
                # Just a path
                self.video_list.addItem(os.path.basename(video))
    
    def on_video_selected(self, row):
        """Handle selection of a video from the list"""
        if row < 0 or row >= len(self.videos):
            return
            
        # Save current classes before switching videos
        if self.current_video_path and self.current_video_path in self.video_classes:
            # Ensure current video's classes are saved in memory
            self.video_classes[self.current_video_path] = self.get_current_classes()
            
        # Get the selected video path
        if isinstance(self.videos[row], dict) and 'path' in self.videos[row]:
            self.current_video_path = self.videos[row]['path']
        else:
            self.current_video_path = self.videos[row]
            
        # Load classes for this video
        self.load_classes()
        
        # Update UI
        self.update_class_table()
    
    def get_current_classes(self):
        """Get a copy of the current classes with properly assigned IDs"""
        classes = []
        for i, cls in enumerate(self.video_classes.get(self.current_video_path, [])):
            classes.append({
                "id": i,
                "name": cls["name"]
            })
        return classes
    
    def load_classes(self):
        """Load classes from classes.csv for the current video"""
        if not self.current_video_path or not self.annotations_dir:
            return
            
        # Check if we've already loaded classes for this video
        if self.current_video_path in self.video_classes:
            return
            
        # Get the video name without extension
        video_name = os.path.splitext(os.path.basename(self.current_video_path))[0]
        
        # Path to classes.csv - updated to include "videos" in the path
        classes_csv_path = os.path.join(self.annotations_dir, "videos", video_name, "classes.csv")
        
        classes = []
        if not os.path.exists(classes_csv_path):
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(classes_csv_path), exist_ok=True)
            
            # Just initialize with unknown class
            classes = [
                {"id": 0, "name": "unknown"}
            ]
        else:
            try:
                with open(classes_csv_path, 'r', newline='') as csvfile:
                    reader = csv.reader(csvfile)
                    next(reader)  # Skip header row
                    for row in reader:
                        if len(row) >= 2:
                            classes.append({
                                "id": int(row[0]),
                                "name": row[1]
                            })
            except Exception as e:
                print(f"Error loading classes from CSV: {str(e)}")
                # Initialize with unknown class on error
                classes = [
                    {"id": 0, "name": "unknown"}
                ]
        
        # Store classes for this video
        self.video_classes[self.current_video_path] = classes
    
    def update_class_table(self):
        """Update the class table with current classes"""
        self.class_table.setRowCount(0)  # Clear table
        
        if not self.current_video_path or self.current_video_path not in self.video_classes:
            return
            
        classes = self.video_classes[self.current_video_path]
        
        # First, ensure IDs are contiguous and sorted
        for i, cls in enumerate(classes):
            cls["id"] = i
            
        for i, cls in enumerate(classes):
            self.class_table.insertRow(i)
            
            # Index column
            index_item = QTableWidgetItem(str(cls["id"]))
            self.class_table.setItem(i, 0, index_item)
            
            # Name column
            name_item = QTableWidgetItem(cls["name"])
            self.class_table.setItem(i, 1, name_item)
    
    def add_class(self):
        """Add a new class to the list"""
        if not self.current_video_path:
            QMessageBox.warning(self, "No Video Selected", 
                              "Please select a video before adding classes.")
            return
            
        if self.current_video_path not in self.video_classes:
            self.video_classes[self.current_video_path] = [{"id": 0, "name": "unknown"}]
            
        classes = self.video_classes[self.current_video_path]
        
        # Get the next available ID - now just use the length as we ensure contiguous IDs
        next_id = len(classes)
            
        # Add new class
        classes.append({
            "id": next_id,
            "name": f"new_class_{next_id}"
        })
        
        # Update table
        self.update_class_table()
        
        # Select the new row
        new_row = len(classes) - 1
        self.class_table.selectRow(new_row)
            
        # Open edit dialog for the new class name
        self.edit_class_name(new_row)
    
    def edit_class_name(self, row):
        """Open a dialog to edit class name"""
        if not self.current_video_path or self.current_video_path not in self.video_classes:
            return
            
        classes = self.video_classes[self.current_video_path]
        
        if row < 0 or row >= len(classes):
            return
            
        current_name = classes[row]["name"]
        text, ok = QInputDialog.getText(
            self,
            "Edit Class Name",
            f"Enter new name for class {classes[row]['id']}:",
            text=current_name
        )
        
        if ok and text:
            classes[row]["name"] = text
            self.update_class_table()
    
    def delete_class(self):
        """Delete the selected class from the list"""
        if not self.current_video_path or self.current_video_path not in self.video_classes:
            return
            
        selected_rows = self.class_table.selectionModel().selectedRows()
        if not selected_rows:
            QMessageBox.warning(self, "No Selection", "Please select a class to delete.")
            return
            
        # Get the selected row
        row = selected_rows[0].row()
        
        classes = self.video_classes[self.current_video_path]
        
        if row < 0 or row >= len(classes):
            return
            
        # Confirm deletion
        class_name = classes[row]["name"]
        if QMessageBox.question(
            self, 
            "Confirm Deletion", 
            f"Are you sure you want to delete the class '{class_name}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        ) == QMessageBox.StandardButton.No:
            return
            
        # Remove the class
        del classes[row]
        
        # Update table
        self.update_class_table()
    
    def accept(self):
        """Save changes and close dialog"""
        if self.annotations_dir:
            # First ensure current video's classes are updated in the dictionary
            if self.current_video_path and self.current_video_path in self.video_classes:
                self.video_classes[self.current_video_path] = self.get_current_classes()
            
            # Save classes for all videos that have been modified
            for video_path in self.video_classes:
                self.save_classes(video_path)
            
            # Show a single success message
            QMessageBox.information(self, "Success", "All class definitions have been saved.")
            
            # Emit signal for current video if available
            if self.current_video_path and self.current_video_path in self.video_classes:
                class_names = [cls["name"] for cls in self.video_classes[self.current_video_path]]
                self.classes_updated.emit(self.current_video_path, class_names)
        
        super().accept()

    def save_classes(self, video_path):
        """Save classes to CSV file for a specific video"""
        if not self.annotations_dir or not video_path or video_path not in self.video_classes:
            return
            
        classes = self.video_classes[video_path]
        
        # Update class IDs based on row position
        for i, cls in enumerate(classes):
            cls["id"] = i
        
        # Get the video name without extension
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        # Path to classes.csv - updated to include "videos" in the path
        classes_csv_path = os.path.join(self.annotations_dir, "videos", video_name, "classes.csv")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(classes_csv_path), exist_ok=True)
        
        try:
            with open(classes_csv_path, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['class_id', 'class_name'])  # Header
                
                # Write classes
                for cls in classes:
                    writer.writerow([cls["id"], cls["name"]])
                    
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save classes for {video_name}: {str(e)}")
