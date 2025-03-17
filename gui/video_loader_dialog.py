from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QListWidget, QLabel,
    QListWidgetItem, QAbstractItemView, QCheckBox, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QPixmap, QImage, QIcon
import os
import cv2
import shutil

class VideoThumbnailItem(QListWidgetItem):
    def __init__(self, video_path, parent=None):
        super().__init__(parent)
        self.video_path = video_path
        self.video_name = os.path.basename(video_path)
        self.setSizeHint(QSize(180, 150))  # Set size for the item
        self.is_selected = False  # Track selection state
        
        # Generate thumbnail
        self.set_thumbnail()
        
        # Set tooltip with video info
        self.set_video_info()
    
    def set_thumbnail(self):
        """Generate a thumbnail for the video"""
        cap = cv2.VideoCapture(self.video_path)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Convert the frame to a thumbnail
                frame = cv2.resize(frame, (160, 120))
                # Convert from BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                # Create QImage and QPixmap
                h, w, ch = frame.shape
                img = QImage(frame.data, w, h, w * ch, QImage.Format.Format_RGB888)
                pixmap = QPixmap.fromImage(img)
                # Set as icon
                self.setIcon(QIcon(pixmap))
            cap.release()
    
    def set_video_info(self):
        """Set tooltip with video information"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if cap.isOpened():
                # Get video properties
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frame_count / fps if fps > 0 else 0
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                
                # Format tooltip
                info = f"Name: {self.video_name}\n"
                info += f"Resolution: {width}x{height}\n"
                info += f"Duration: {duration:.2f} seconds\n"
                info += f"FPS: {fps:.2f}\n"
                info += f"Frames: {frame_count}"
                
                self.setToolTip(info)
                
                # Also set the text to the video name
                self.setText(self.video_name)
                
                cap.release()
        except Exception as e:
            self.setToolTip(f"Error loading video info: {str(e)}")
            self.setText(self.video_name)

class VideoLoaderDialog(QDialog):
    # Removed the video_selected signal
    videos_selected = pyqtSignal(list, list)  # paths, display_names (multiple videos)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Video")
        self.resize(800, 500)
        self.last_selected_item = None  # Track the last selected item
        self.setup_ui()
        self.load_videos()
    
    def setup_ui(self):
        # Main layout
        main_layout = QVBoxLayout(self)
        
        # Title label
        title = QLabel("Select Video(s) to Load")
        title.setStyleSheet("font-size: 16pt; font-weight: bold; margin-bottom: 10px;")
        main_layout.addWidget(title)
        
        # Multiple selection checkbox
        self.multi_select_checkbox = QCheckBox("Allow Multiple Selection")
        self.multi_select_checkbox.toggled.connect(self.toggle_multiple_selection)
        main_layout.addWidget(self.multi_select_checkbox)
        
        # Videos grid
        self.videos_list = QListWidget()
        self.videos_list.setViewMode(QListWidget.ViewMode.IconMode)
        self.videos_list.setIconSize(QSize(160, 120))
        self.videos_list.setResizeMode(QListWidget.ResizeMode.Adjust)
        self.videos_list.setSpacing(10)
        self.videos_list.setMovement(QListWidget.Movement.Static)  # Items don't move
        self.videos_list.itemDoubleClicked.connect(self.on_video_double_clicked)
        self.videos_list.itemClicked.connect(self.on_video_clicked)
        # Install event filter for handling clicks on empty space
        self.videos_list.installEventFilter(self)
        main_layout.addWidget(self.videos_list)
        
        # Buttons layout
        buttons_layout = QHBoxLayout()
        
        # Refresh button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.clicked.connect(self.load_videos)
        buttons_layout.addWidget(self.refresh_btn)
        
        # Clear All button
        self.clear_all_btn = QPushButton("Clear All Videos")
        self.clear_all_btn.clicked.connect(self.clear_all_videos)
        self.clear_all_btn.setStyleSheet("background-color: #8B0000; color: white;")  # Dark red background
        buttons_layout.addWidget(self.clear_all_btn)
        
        # Spacer
        buttons_layout.addStretch()
        
        # Load button
        self.load_btn = QPushButton("Load Selected")
        self.load_btn.clicked.connect(self.load_selected_video)
        buttons_layout.addWidget(self.load_btn)
        
        # Cancel button
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(self.cancel_btn)
        
        main_layout.addLayout(buttons_layout)
        
        # Apply styles
        self.apply_styles()
    
    def apply_styles(self):
        """Apply modern styling to the dialog"""
        # Button style
        button_style = """
            QPushButton {
                color: white;
                background-color: #3d3d3d;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #4a4a4a;
            }
            QPushButton:pressed {
                background-color: #2d2d2d;
            }
        """
        
        # List style
        list_style = """
            QListWidget {
                background-color: #2d2d2d;
                border: 1px solid #444444;
                border-radius: 4px;
                padding: 5px;
            }
            QListWidget::item {
                background-color: #3a3a3a;
                border: 1px solid #555555;
                border-radius: 5px;
                margin: 5px;
            }
            QListWidget::item:selected {
                background-color: #4a6b8a;
                border: 1px solid #6080a0;
            }
            QListWidget::item:hover:!selected {
                background-color: #454545;
            }
        """
        
        # Apply styles
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)
        
        self.videos_list.setStyleSheet(list_style)
    
    def toggle_multiple_selection(self, enabled):
        """Toggle between single and multiple selection modes"""
        if enabled:
            self.videos_list.setSelectionMode(QAbstractItemView.SelectionMode.MultiSelection)
        else:
            self.videos_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
            
        # Clear selections when changing mode
        self.videos_list.clearSelection()
        for idx in range(self.videos_list.count()):
            item = self.videos_list.item(idx)
            if hasattr(item, 'is_selected'):
                item.is_selected = False
    
    def load_videos(self):
        """Load all videos from datasets/videos directory"""
        self.videos_list.clear()
        videos_dir = os.path.join("datasets", "videos")
        
        if not os.path.exists(videos_dir):
            os.makedirs(videos_dir, exist_ok=True)
            return
        
        # Get all video files
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv']
        for file in os.listdir(videos_dir):
            file_path = os.path.join(videos_dir, file)
            if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in video_extensions):
                # Create a custom list item with thumbnail
                item = VideoThumbnailItem(file_path)
                self.videos_list.addItem(item)
    
    def eventFilter(self, source, event):
        """Handle events for the list widget"""
        if (source is self.videos_list and 
            event.type() == event.Type.MouseButtonPress):
            # Get the item at the position of the mouse click
            print('help')
            item = self.videos_list.itemAt(event.position().toPoint())
            # If clicked on empty space
            if not item:
                # Clear all selections
                for idx in range(self.videos_list.count()):
                    curr_item = self.videos_list.item(idx)
                    curr_item.is_selected = False
                self.last_selected_item = None
                # Let the list widget clear its visual selection
                self.videos_list.clearSelection()
        return super().eventFilter(source, event)
    
    def on_video_clicked(self, item):
        """Handle click on a video to toggle selection"""
        if not self.multi_select_checkbox.isChecked():
            # In single selection mode, clear previous selections
            for idx in range(self.videos_list.count()):
                curr_item = self.videos_list.item(idx)
                if curr_item != item and hasattr(curr_item, 'is_selected') and curr_item.is_selected:
                    curr_item.is_selected = False
                    curr_item.setSelected(False)
        
        # Toggle selection on the clicked item
        if hasattr(item, 'is_selected'):
            item.is_selected = not item.is_selected
            item.setSelected(item.is_selected)
            self.last_selected_item = item if item.is_selected else None
    
    def on_video_double_clicked(self, item):
        """Handle double-click on video item"""
        # Treat double-click as selecting a single video
        self.videos_selected.emit([item.video_path], [item.video_name])
        self.accept()
    
    def load_selected_video(self):
        """Load the selected video(s)"""
        selected_items = [
            item for idx in range(self.videos_list.count())
            if (item := self.videos_list.item(idx)).is_selected
        ]
        
        if selected_items:
            # Emit all selected videos using the unified signal
            paths = [item.video_path for item in selected_items]
            names = [item.video_name for item in selected_items]
            self.videos_selected.emit(paths, names)
            self.accept()
    
    def clear_all_videos(self):
        """Clear all videos and their associated files after user confirmation"""
        # Show confirmation dialog
        confirm = QMessageBox.question(
            self, 
            "Confirm Deletion",
            "This will permanently delete ALL videos and their associated annotation and configuration files.\n\n"
            "Are you sure you want to continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        
        if confirm == QMessageBox.StandardButton.Yes:
            # Directories to clear
            dirs_to_clear = [
                os.path.join("datasets", "videos"),
                os.path.join("datasets", "annotations", "videos"),
                os.path.join("datasets", "video_configs")
            ]
            
            success = True
            errors = []
            
            # Clear each directory
            for directory in dirs_to_clear:
                if os.path.exists(directory):
                    try:
                        # Delete all files but keep the directory structure
                        for file_name in os.listdir(directory):
                            file_path = os.path.join(directory, file_name)
                            if os.path.isfile(file_path):
                                try:
                                    os.remove(file_path)
                                except Exception as e:
                                    success = False
                                    errors.append(f"Failed to delete {file_path}: {str(e)}")
                    except Exception as e:
                        success = False
                        errors.append(f"Error accessing directory {directory}: {str(e)}")
            
            # Refresh the video list
            self.load_videos()
            
            # Show result message
            if success:
                QMessageBox.information(self, "Success", "All videos and associated files have been cleared.")
            else:
                error_msg = "Some errors occurred while clearing files:\n" + "\n".join(errors)
                QMessageBox.warning(self, "Warning", error_msg)
