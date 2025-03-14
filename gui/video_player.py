import cv2
import os
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (QLabel, QMessageBox, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QSlider, QLineEdit, QStyle, QSizePolicy, QFrame)

class VideoPlayer(QWidget):
    frame_changed = pyqtSignal(int)  # Signal to notify frame changes
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrameSlot)
        self._playing = False
        self.frame_rate = 30  # Default frame rate
        self.total_frames = 0
        self.current_frame_idx = 0
        self.current_frame = None
        self.skip_frames = 10  # Default frame skip amount
        self.video_source_type = "No video"  # Track the source of the video
        self.setupUI()
        self._apply_styles()
    
    def _invert_icon(self, standard_pixmap):
        """Inverts the colors of a standard icon for better visibility on dark backgrounds"""
        pixmap = self.style().standardPixmap(standard_pixmap)
        image = pixmap.toImage()
        image.invertPixels()
        return QIcon(QPixmap.fromImage(image))
        
    def setupUI(self):
        # Main layout:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Video display
        self.display = QLabel()
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display.setMinimumSize(640, 480)
        self.display.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.display.setFrameShape(QFrame.Shape.Box)
        self.display.setText("No video loaded")
        main_layout.addWidget(self.display)
        
        # Control layout
        control_layout = QVBoxLayout()
        
        # Frame counter display
        self.frame_counter = QLabel("Frame: 0 / 0")
        self.frame_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.frame_counter)
        
        # Slider for video navigation
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.seek_position)
        self.position_slider.sliderPressed.connect(self.slider_pressed)
        self.position_slider.sliderReleased.connect(self.slider_released)
        self.position_slider.valueChanged.connect(self.slider_value_changed)
        control_layout.addWidget(self.position_slider)
        
        # Set consistent button width and style
        button_width = 80
        text_width = 80
        button_spacing = 10
        
        # FIRST ROW: Play button, frame rate controls, and go to frame controls
        first_row_layout = QHBoxLayout()
        first_row_layout.addStretch()  # Add spacer for centering
        
        # Play/Pause button with icon
        self.play_button = QPushButton()
        self.play_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setToolTip("Play/Pause")
        self.play_button.setFixedWidth(button_width)
        first_row_layout.addWidget(self.play_button)
        
        # Add vertical separator after play button
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.VLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        first_row_layout.addWidget(separator1)
        
        # Frame rate controls
        first_row_layout.addWidget(QLabel("Frame Rate:"))
        self.rate_text = QLineEdit()
        self.rate_text.setText(str(int(self.frame_rate)))
        self.rate_text.setFixedWidth(text_width)
        self.rate_text.textChanged.connect(self.update_frame_rate)
        self.rate_text.setToolTip("Frames per second (FPS)")
        first_row_layout.addWidget(self.rate_text)
        first_row_layout.addSpacing(button_spacing)
        
        # Apply frame rate button
        self.apply_rate_button = QPushButton("Apply")
        self.apply_rate_button.setFixedWidth(button_width)
        self.apply_rate_button.clicked.connect(self.apply_frame_rate)
        self.apply_rate_button.setToolTip("Apply frame rate change")
        first_row_layout.addWidget(self.apply_rate_button)
        
        # Add vertical separator after frame rate controls
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        first_row_layout.addWidget(separator2)
        
        # Go to specific frame functionality
        self.frame_input = QLineEdit()
        self.frame_input.setText("0")
        self.frame_input.setToolTip("Enter frame number")
        self.frame_input.setFixedWidth(text_width)
        first_row_layout.addWidget(self.frame_input)
        first_row_layout.addSpacing(button_spacing)
        
        self.goto_button = QPushButton("Go to Frame")
        self.goto_button.clicked.connect(self.goto_frame)
        self.goto_button.setToolTip("Jump to specified frame")
        self.goto_button.setFixedWidth(int(button_width * 1.5))
        first_row_layout.addWidget(self.goto_button)
        
        first_row_layout.addStretch()  # Add spacer for centering
        control_layout.addLayout(first_row_layout)
        
        # SECOND ROW: Previous, next, skip backward, and skip forward buttons
        second_row_layout = QHBoxLayout()
        second_row_layout.addStretch()  # Add spacer for centering
        
        # Previous frame button
        self.prev_button = QPushButton()
        self.prev_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setToolTip("Previous Frame")
        self.prev_button.setFixedWidth(button_width)
        second_row_layout.addWidget(self.prev_button)
        second_row_layout.addSpacing(button_spacing)
        
        # Next frame button
        self.next_button = QPushButton()
        self.next_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setToolTip("Next Frame")
        self.next_button.setFixedWidth(button_width)
        second_row_layout.addWidget(self.next_button)
        
        # Add vertical separator after previous/next buttons
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.VLine)
        separator3.setFrameShadow(QFrame.Shadow.Sunken)
        second_row_layout.addWidget(separator3)
        
        # Frame skip amount input - MOVED to between prev/next and skip buttons
        second_row_layout.addWidget(QLabel("Skip Frames:"))
        self.skip_text = QLineEdit()
        self.skip_text.setText(str(self.skip_frames))
        self.skip_text.textChanged.connect(self.update_skip_amount)
        self.skip_text.setToolTip("Number of frames to skip")
        self.skip_text.setFixedWidth(text_width)
        second_row_layout.addWidget(self.skip_text)
        
        # Skip backward button
        self.skip_backward = QPushButton()
        self.skip_backward.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaSeekBackward))
        self.skip_backward.clicked.connect(self.skip_backward_frames)
        self.skip_backward.setToolTip("Skip Backward")
        self.skip_backward.setFixedWidth(button_width)
        second_row_layout.addWidget(self.skip_backward)
        second_row_layout.addSpacing(button_spacing)
        
        # Skip forward button
        self.skip_forward = QPushButton()
        self.skip_forward.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaSeekForward))
        self.skip_forward.clicked.connect(self.skip_forward_frames)
        self.skip_forward.setToolTip("Skip Forward")
        self.skip_forward.setFixedWidth(button_width)
        second_row_layout.addWidget(self.skip_forward)
        
        second_row_layout.addStretch()  # Add spacer for centering
        control_layout.addLayout(second_row_layout)
        
        # Video information display
        self.info_label = QLabel("No video loaded")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.info_label)
        
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)
        
    def _apply_styles(self):
        """Apply consistent styling to all buttons and controls"""
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
        
        # Spinbox and Input style
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
        
        # Style for sliders
        slider_style = """
            QSlider::groove:horizontal {
                height: 8px;
                background: #2d2d2d;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #3d3d3d;
                border: none;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #4a4a4a;
            }
            QSlider::handle:horizontal:disabled {
                background: #555555;
            }
            QSlider:disabled {
                opacity: 0.7;
            }
        """
        
        # Apply styles to all buttons
        for button in self.findChildren(QPushButton):
            button.setStyleSheet(button_style)
        
        # Apply style to spinboxes and text inputs
        for line_edit in self.findChildren(QLineEdit):
            line_edit.setStyleSheet(text_input_style)
        
        # Apply style to slider only
        self.position_slider.setStyleSheet(slider_style)
        
    def load_video(self, video_path, source_type="Original Video"):
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(None, "Error", f"Could not open video file: {video_path}")
                return False
                
            # Get video properties
            self.current_frame_idx = 0
            self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
            if self.frame_rate <= 0:
                self.frame_rate = 30  # Fallback frame rate
                
            # Update UI elements
            self.position_slider.setRange(0, self.total_frames - 1)
            self.position_slider.setValue(0)
            self.frame_input.setText("0")  # Set text instead of value for QLineEdit
            self.frame_counter.setText(f"Frame: 0 / {self.total_frames}")
            
            # Set video source type and update info
            self.video_source_type = source_type
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_info = f"Source: {self.video_source_type} | Resolution: {width}x{height} | FPS: {self.frame_rate:.2f} | Frames: {self.total_frames}"
            self.info_label.setText(self.video_info)
            
            # Load first frame
            ret, frame = self.cap.read()
            if ret:
                self.set_image(frame)
                self.current_frame = frame
                return True
            else:
                QMessageBox.critical(None, "Error", "Could not read frames from video")
                return False
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading video: {str(e)}")
            return False
    
    def set_video_source_type(self, source_type):
        """Update the video source type information"""
        self.video_source_type = source_type
        # Update the info label if video is loaded
        if hasattr(self, 'video_info') and self.cap is not None:
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_info = f"Source: {self.video_source_type} | Resolution: {width}x{height} | FPS: {self.frame_rate:.2f} | Frames: {self.total_frames}"
            self.info_label.setText(self.video_info)
    
    def load_image(self, image_path, source_type="Original Image"):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                QMessageBox.critical(None, "Error", f"Could not open image file: {image_path}")
                return False
                        
            self.current_frame = frame
            self.set_image(frame)
            self.video_source_type = source_type
            self.info_label.setText(f"Source: {self.video_source_type} | Image: {os.path.basename(image_path)}")
            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading image: {str(e)}")
            return False
    
    def nextFrameSlot(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
        try:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.current_frame_idx = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                self.position_slider.setValue(self.current_frame_idx)
                self.set_image(frame)
                self.update_frame_counter()
                self.frame_changed.emit(self.current_frame_idx)
            else:
                # End of video
                self.pause()
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.current_frame_idx = 0
                self.position_slider.setValue(0)
                self.update_frame_counter()
        except Exception as e:
            print(f"Error reading next frame: {str(e)}")
    
    def set_image(self, frame):
        if frame is None:
            return
            
        try:
            # Convert BGR to RGB and set image in QLabel
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = frame_rgb.shape
            bytes_per_line = ch * w
            qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            
            # Check if display exists before trying to update it
            if hasattr(self, 'display') and self.display is not None:
                self.display.setPixmap(pixmap.scaled(self.display.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                           Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            print(f"Error displaying frame: {str(e)}")
    
    def set_frame(self, frame):
        self.current_frame = frame
        self.set_image(frame)
    
    def toggle_playback(self):
        if self._playing:
            self.pause()
        else:
            self.play()
    
    def update_frame_rate(self, text):
        """Update the frame rate setting"""
        try:
            if text.strip():  # Check if the text is not empty
                value = float(text)
                if value > 0:  # Ensure positive value
                    self.frame_rate = value
                    # If already playing, restart with new frame rate
                    if self._playing:
                        self.pause()
                        self.play()
        except ValueError:
            # If text is not a valid number, keep the previous value
            self.rate_text.setText(str(int(self.frame_rate)))
    
    def apply_frame_rate(self):
        """Apply the current frame rate setting"""
        try:
            value = float(self.rate_text.text())
            if value > 0:
                self.frame_rate = value
                # Update timer if currently playing
                if self._playing:
                    self.pause()
                    self.play()
                QMessageBox.information(self, "Frame Rate", f"Frame rate set to {self.frame_rate} FPS")
            else:
                QMessageBox.warning(self, "Invalid Value", "Frame rate must be greater than 0")
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number for frame rate")
    
    def play(self):
        if self.cap is not None and self.cap.isOpened():
            # Calculate milliseconds per frame for smoother playback
            ms_per_frame = int(1000 / self.frame_rate)
            self.timer.start(ms_per_frame)
            self._playing = True
            self.play_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            QMessageBox.warning(None, "Playback Error", "No valid video is loaded")
    
    def pause(self):
        self.timer.stop()
        self._playing = False
        self.play_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaPlay))
    
    def is_playing(self):
        return self._playing
    
    def get_current_frame(self):
        return self.current_frame
    
    def get_current_frame_idx(self):
        return self.current_frame_idx
    
    def get_total_frames(self):
        return self.total_frames
    
    def seek(self, frame_idx):
        if self.cap is None:
            return False
            
        try:
            frame_idx = max(0, min(frame_idx, self.total_frames - 1))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.current_frame_idx = frame_idx
                self.set_image(frame)
                self.position_slider.setValue(frame_idx)
                self.update_frame_counter()
                self.frame_changed.emit(frame_idx)
                return True
            else:
                print(f"Failed to read frame at index {frame_idx}")
        except Exception as e:
            print(f"Error seeking to frame {frame_idx}: {str(e)}")
        return False
    
    def seek_position(self, position):
        self.seek(position)
    
    def slider_pressed(self):
        """Called when slider is pressed to temporarily pause playback"""
        if self._playing:
            self.timer.stop()
    
    def slider_released(self):
        """Called when slider is released to resume playback if it was playing"""
        if self._playing:
            self.timer.start()
    
    def next_frame(self):
        """Move to the next frame"""
        if self.cap is not None:
            self.pause()  # Pause playback
            next_frame = self.current_frame_idx + 1
            self.seek(next_frame)
    
    def prev_frame(self):
        """Move to the previous frame"""
        if self.cap is not None:
            self.pause()  # Pause playback
            prev_frame = self.current_frame_idx - 1
            self.seek(prev_frame)
    
    def update_skip_amount(self, text):
        """Update the number of frames to skip"""
        try:
            if text.strip():  # Check if the text is not empty
                value = int(text)
                if value > 0:  # Ensure positive value
                    self.skip_frames = value
        except ValueError:
            # If text is not a valid integer, keep the previous value
            self.skip_text.setText(str(self.skip_frames))
    
    def skip_forward_frames(self):
        """Skip forward by the specified number of frames"""
        if self.cap is not None:
            self.pause()
            target_frame = min(self.current_frame_idx + self.skip_frames, self.total_frames - 1)
            self.seek(target_frame)
    
    def skip_backward_frames(self):
        """Skip backward by the specified number of frames"""
        if self.cap is not None:
            self.pause()
            target_frame = max(0, self.current_frame_idx - self.skip_frames)
            self.seek(target_frame)
    
    def resizeEvent(self, event):
        if self.current_frame is not None:
            self.set_image(self.current_frame)
        super().resizeEvent(event)
    
    def update_frame_counter(self):
        """Update the frame counter label"""
        self.frame_counter.setText(f"Frame: {self.current_frame_idx} / {self.total_frames}")
    
    def slider_value_changed(self, value):
        """Called when the slider value changes (including from clicks)"""
        if not self.position_slider.isSliderDown():
            self.seek(value)
    
    def goto_frame(self):
        """Go to the frame specified in the frame input box"""
        try:
            frame_num = int(self.frame_input.text())
            if self.cap is not None:
                # Ensure frame number is within valid range
                frame_num = max(0, min(frame_num, self.total_frames - 1))
                self.pause()
                self.seek(frame_num)
            else:
                QMessageBox.warning(None, "No Video", "Please load a video first")
        except ValueError:
            QMessageBox.warning(None, "Invalid Input", "Please enter a valid frame number")
    
    def __del__(self):
        """Destructor to ensure resources are properly released"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()
