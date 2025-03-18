import cv2
import os
import time
from PyQt6.QtCore import QTimer, Qt, pyqtSignal, QElapsedTimer
from PyQt6.QtGui import QImage, QPixmap, QIcon, QPainter, QPen, QColor
from PyQt6.QtWidgets import (QLabel, QMessageBox, QWidget, QVBoxLayout, 
                           QHBoxLayout, QPushButton, QSlider, QLineEdit, QStyle, QSizePolicy, QFrame,
                           QComboBox, QProgressBar)

# Import our video buffer system
from .video_buffer import VideoBuffer

class VideoPlayer(QWidget):
    frame_changed = pyqtSignal(int)  # Signal to notify frame changes
    video_changed = pyqtSignal(int)  # Signal to notify video changes
    
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
        self.annotations = []  # Store annotations for current frame
        
        # Multiple video support
        self.videos = []  # List of dicts with video info: {path, name, cap, frame_idx, total_frames}
        self.current_video_index = -1  # Index of currently selected video
        
        # Remove performance monitoring code
        self.frame_times = []
        self.dropped_frames = 0
        self.last_dropped_warning = 0
        self.adaptive_mode = True  # Keep adaptive mode for smoother playback
        
        # Enhanced frame timing and framerate management
        self.target_frame_time = 33.33  # Target time per frame in ms (30 fps)
        self.actual_frame_time = 33.33  # Actual time per frame in ms
        self.playback_speed = 1.0  # Playback speed multiplier
        self.high_adaptive_fps_mode = False  # Toggle for high adaptive FPS mode
        self.adaptive_fps_offset = 1000.0  # Custom offset for high adaptive FPS mode (in ms)
        
        # Improved precise timing tools using perf_counter instead of QElapsedTimer
        self.last_frame_time = None  # Will store the last frame timestamp
        self.ewma_factor = 0.4  # Increased from 0.2 for faster adaptation
        self.ewma_frame_time = 33.33  # Initial estimate for frame processing time
        self.base_adjustment_rate = 1  # Significantly higher adjustment rate (was 0.5)
        self.adaptive_adjustment = True  # Enable adaptive adjustment rate
        self.frame_time_history = []  # Track recent frame times for stability detection
        self.history_size = 5  # Number of frame times to track for stability
        self.stable_threshold = 1.0  # ms threshold for considering timing stable
        
        # Initialize video buffer system
        self.video_buffer = VideoBuffer(self)
        self.video_buffer.buffer_status_updated.connect(self.on_buffer_status_updated)
        self.video_buffer.buffer_ready.connect(self.on_buffer_ready)
        self.video_buffer.end_of_file_reached.connect(self.on_end_of_file_reached)  # Single EOF signal connection
        self.buffer_ready = False
        self.eof_reached = False  # Track EOF state
        
        self.setupUI()
        self._apply_styles()
        
        # Remove the separate stats timer
    
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
        
        # Buffer status bar (new)
        self.buffer_status = QProgressBar()
        self.buffer_status.setRange(0, 100)
        self.buffer_status.setValue(0)
        self.buffer_status.setFormat("Buffer: %p%")
        self.buffer_status.setTextVisible(True)
        main_layout.addWidget(self.buffer_status)
        
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
        first_row_layout.addSpacing(button_spacing)  # Add spacing before separator
        separator1 = QFrame()
        separator1.setFrameShape(QFrame.Shape.VLine)
        separator1.setFrameShadow(QFrame.Shadow.Sunken)
        first_row_layout.addWidget(separator1)
        first_row_layout.addSpacing(button_spacing)  # Add spacing after separator
        
        # Frame rate controls
        first_row_layout.addWidget(QLabel("Frame Rate:"))
        self.rate_text = QLineEdit()
        self.rate_text.setText(str(int(self.frame_rate)))
        self.rate_text.setFixedWidth(text_width)
        self.rate_text.textChanged.connect(self.update_frame_rate)
        self.rate_text.setToolTip("Frames per second (FPS)")
        first_row_layout.addWidget(self.rate_text)
        
        # Apply frame rate button
        self.apply_rate_button = QPushButton("Apply")
        self.apply_rate_button.setFixedWidth(button_width)
        self.apply_rate_button.clicked.connect(self.apply_frame_rate)
        self.apply_rate_button.setToolTip("Apply frame rate change")
        first_row_layout.addWidget(self.apply_rate_button)
        
        # Add vertical separator after frame rate controls
        first_row_layout.addSpacing(button_spacing)  # Add spacing before separator
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.VLine)
        separator2.setFrameShadow(QFrame.Shadow.Sunken)
        first_row_layout.addWidget(separator2)
        first_row_layout.addSpacing(button_spacing)  # Add spacing after separator
        
        # Go to specific frame functionality
        first_row_layout.addWidget(QLabel("Frame:"))
        self.frame_input = QLineEdit()
        self.frame_input.setText("0")
        self.frame_input.setToolTip("Enter frame number")
        self.frame_input.setFixedWidth(text_width)
        first_row_layout.addWidget(self.frame_input)
        
        self.goto_button = QPushButton("Go to Frame")
        self.goto_button.clicked.connect(self.goto_frame)
        self.goto_button.setToolTip("Jump to specified frame")
        self.goto_button.setFixedWidth(button_width)
        first_row_layout.addWidget(self.goto_button)
        
        # High Adaptive FPS Mode toggle (renamed from high performance mode)
        self.high_perf_button = QPushButton("Adaptive: Normal")
        self.high_perf_button.setFixedWidth(button_width + 30)
        self.high_perf_button.clicked.connect(self.toggle_high_adaptive_fps)
        self.high_perf_button.setToolTip("Toggle between normal and high adaptive FPS modes")
        first_row_layout.addWidget(self.high_perf_button)
        
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
        
        # Next frame button
        self.next_button = QPushButton()
        self.next_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setToolTip("Next Frame")
        self.next_button.setFixedWidth(button_width)
        second_row_layout.addWidget(self.next_button)
        
        # Add vertical separator after previous/next buttons
        second_row_layout.addSpacing(button_spacing)  # Add spacing before separator
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.VLine)
        separator3.setFrameShadow(QFrame.Shadow.Sunken)
        second_row_layout.addWidget(separator3)
        second_row_layout.addSpacing(button_spacing)  # Add spacing after separator
        
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
        
        # Skip forward button
        self.skip_forward = QPushButton()
        self.skip_forward.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaSeekForward))
        self.skip_forward.clicked.connect(self.skip_forward_frames)
        self.skip_forward.setToolTip("Skip Forward")
        self.skip_forward.setFixedWidth(button_width)
        second_row_layout.addWidget(self.skip_forward)
        
        second_row_layout.addStretch()  # Add spacer for centering
        control_layout.addLayout(second_row_layout)
        
        # THIRD ROW: Video navigation controls (NEW)
        third_row_layout = QHBoxLayout()
        third_row_layout.addStretch()  # Add spacer for centering
        
        # Previous video button
        self.prev_video_button = QPushButton()
        self.prev_video_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_ArrowLeft))
        self.prev_video_button.clicked.connect(self.prev_video)
        self.prev_video_button.setToolTip("Previous Video")
        self.prev_video_button.setFixedWidth(button_width)
        third_row_layout.addWidget(self.prev_video_button)
        
        # Video selector dropdown
        third_row_layout.addWidget(QLabel("Video:"))
        self.video_selector = QComboBox()
        self.video_selector.setMinimumWidth(250)
        self.video_selector.currentIndexChanged.connect(self.on_video_selected)
        self.video_selector.setToolTip("Select video")
        third_row_layout.addWidget(self.video_selector)
        
        # Next video button
        self.next_video_button = QPushButton()
        self.next_video_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_ArrowRight))
        self.next_video_button.clicked.connect(self.next_video)
        self.next_video_button.setToolTip("Next Video")
        self.next_video_button.setFixedWidth(button_width)
        third_row_layout.addWidget(self.next_video_button)
        
        third_row_layout.addStretch()  # Add spacer for centering
        control_layout.addLayout(third_row_layout)
        
        # Video information display
        self.info_label = QLabel("No video loaded")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        control_layout.addWidget(self.info_label)
        
        # Remove the performance statistics display
        # self.performance_label = QLabel("FPS: 0.0 | Buffer: 0% | Dropped: 0")
        # self.performance_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # control_layout.addWidget(self.performance_label)
        
        main_layout.addLayout(control_layout)
        self.setLayout(main_layout)

        # Initially disable video navigation buttons
        self.update_video_navigation_buttons()
        
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
                background: #888888;  /* Changed from #3d3d3d to make it lighter */
                border: none;
                width: 16px;
                height: 16px;
                margin: -4px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #aaaaaa;  /* Made hover state lighter too */
                border: 1px solid #bbbbbb;  /* Added border for better visibility */
            }
            QSlider::handle:horizontal:disabled {
                background: #555555;
            }
            QSlider:disabled {
                opacity: 0.7;
            }
        """
        
        # Style for combobox (new)
        combobox_style = """
            QComboBox {
                color: white;
                background-color: #3d3d3d;
                border: 1px solid #555555;
                border-radius: 2px;
                padding: 1px 18px 1px 3px;
            }
            QComboBox:hover {
                background-color: #4a4a4a;
            }
            QComboBox:disabled {
                background-color: #555555;
                color: #888888;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: top right;
                width: 15px;
                border-left-width: 1px;
                border-left-color: #555555;
                border-left-style: solid;
            }
            QComboBox QAbstractItemView {
                color: white;
                background-color: #3d3d3d;
                selection-background-color: #4a6b8a;
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
        
        # Apply style to combobox
        self.video_selector.setStyleSheet(combobox_style)
        
    def load_video(self, video_path, source_type="Original Video"):
        """Load a video file and add it to the video list"""
        return self.add_video(video_path, source_type)
    
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
        """Display the next frame from the buffer"""
        if self.cap is None or not self.cap.isOpened() or self.current_video_index < 0:
            return
            
        # Start frame timing with high precision timer
        start_time = time.perf_counter()
            
        try:
            # Get frame from buffer
            expected_frame = self.current_frame_idx + 1
            
            # Ensure we don't try to go past the last valid frame
            last_valid_frame = self.total_frames - 1
            if expected_frame > last_valid_frame:
                self.handle_video_end()
                return
                
            frame_idx, frame = self.video_buffer.get_frame(expected_frame)
            
            if frame is not None:
                # We got a valid frame
                self.current_frame = frame
                self.current_frame_idx = frame_idx
                
                # Update the current frame index in the videos list
                if self.current_video_index >= 0 and self.current_video_index < len(self.videos):
                    self.videos[self.current_video_index]["frame_idx"] = self.current_frame_idx
                    self.videos[self.current_video_index]["current_frame"] = frame
                
                # Update slider without triggering seek events
                self.position_slider.blockSignals(True)
                self.position_slider.setValue(self.current_frame_idx)
                self.position_slider.blockSignals(False)
                
                # Display the frame
                self.set_image(frame)
                self.update_frame_counter()
                self.frame_changed.emit(self.current_frame_idx)
            else:
                # Buffer is empty - check if EOF reached
                if self.eof_reached:
                    # This is the KEY CONDITION: buffer is empty AND EOF flag is set
                    self.handle_video_end()
                    return
                
                # Buffer is empty but EOF not reached, so we're just waiting for more frames
                if not self.video_buffer.is_buffer_ready():
                    self.display.setText("Buffering...")
            
            # Highly responsive adaptive frame rate handling using high-precision performance counter
            if self.adaptive_mode and self.last_frame_time is not None:
                current_time = time.perf_counter()
                # Get elapsed time in milliseconds with high precision
                elapsed_ms = (current_time - self.last_frame_time) * 1000
                self.last_frame_time = current_time
                
                # Track frame times for stability analysis
                self.frame_time_history.append(elapsed_ms)
                if len(self.frame_time_history) > self.history_size:
                    self.frame_time_history.pop(0)
                
                # Update exponential weighted moving average with higher weight for faster adaptation
                self.ewma_frame_time = (self.ewma_factor * elapsed_ms) + ((1 - self.ewma_factor) * self.ewma_frame_time)
                
                # Calculate error between target and actual frame time
                error = self.ewma_frame_time - self.target_frame_time
                
                # Apply additional offset in high adaptive FPS mode to make corrections more aggressive
                if self.high_adaptive_fps_mode:
                    if error > 0:  # Processing is too slow, increase the error to make adjustment more aggressive
                        error = error + self.adaptive_fps_offset
                    elif error < 0:  # Processing is too fast, make negative error more negative
                        error = error - self.adaptive_fps_offset
                
                # Determine if the timing is stable
                is_stable = len(self.frame_time_history) >= 3 and abs(max(self.frame_time_history) - min(self.frame_time_history)) < self.stable_threshold
                
                # Calculate adaptive adjustment rate based on error magnitude
                adjustment_rate = self.base_adjustment_rate
                if self.adaptive_adjustment:
                    # Make adjustment more aggressive for larger errors
                    error_magnitude = abs(error) / self.target_frame_time
                    if error_magnitude > 0.5:  # Error > 50% of target
                        adjustment_rate = min(1.5, adjustment_rate * 2)  # Much more aggressive
                    elif error_magnitude > 0.2:  # Error > 20% of target
                        adjustment_rate = min(1.2, adjustment_rate * 1.5)  # More aggressive
                    elif is_stable:
                        adjustment_rate = adjustment_rate * 0.8  # More conservative when stable
                
                # Calculate adjustment based on error with adaptive rate
                adjustment = error * adjustment_rate
                
                # - If processing is slow (positive error), DECREASE interval to compensate
                # - If processing is fast (negative error), INCREASE interval to maintain target frame rate
                current_interval = self.timer.interval()
                new_interval = current_interval - adjustment
                
                # Handle extreme cases with immediate correction
                if new_interval < 1 or new_interval > 1000:
                    # Reset to target if we're way off
                    new_interval = self.target_frame_time
                else:
                    # Ensure reasonable minimum value
                    new_interval = max(1.0, new_interval)
                
                self.timer.setInterval(int(new_interval))
        
        except Exception as e:
            print(f"Error in nextFrameSlot: {str(e)}")
    
    def handle_video_end(self):
        """Centralized method to handle when video truly ends (EOF reached and buffer empty)"""
        # Pause playback
        self.pause()
        
        # Ensure position is at last valid frame
        last_frame = self.total_frames - 1
        if self.current_frame_idx != last_frame:
            self.current_frame_idx = last_frame
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(last_frame)
            self.position_slider.blockSignals(False)
            self.update_frame_counter()
        
        # Reset EOF flag for next playback
        self.eof_reached = False
        
        print(f"Video ended at frame {self.current_frame_idx} (total frames: {self.total_frames})")

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
        """Set the current frame to display, with optional annotations"""
        self.current_frame = frame.copy()
        self.set_image(self.current_frame)
    
    def set_image_with_annotations(self, frame, annotations=None, show_labels=True):
        """Set image with optional annotations overlay"""
        if frame is None:
            return
            
        # Make a copy of the frame to draw on
        display_frame = frame.copy()
        
        # Draw annotations if provided
        if annotations:
            for annotation in annotations:
                if 'bbox_abs' in annotation:
                    x1, y1, x2, y2 = annotation['bbox_abs']
                    label = annotation.get('class', '-1')
                    confidence = annotation.get('confidence', 0.0)
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw label and confidence only if show_labels is True
                    if show_labels and label != '-' and label != '-1':
                        text = f"{label}"
                        if confidence > 0:
                            text += f" ({confidence:.2f})"
                        cv2.putText(display_frame, text, (x1, y1-10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # Display the annotated frame
        self.set_image(display_frame)
    
    def annotate_current_frame(self, annotations):
        """Store annotations for current frame and update display"""
        if self.current_frame is not None:
            self.annotations = annotations
            self.set_image_with_annotations(self.current_frame, self.annotations)
    
    def clear_annotations(self):
        """Clear annotations and redisplay the current frame"""
        self.annotations = []
        if self.current_frame is not None:
            self.set_image(self.current_frame)
    
    def capture_frame(self):
        """Capture the current frame from the video"""
        if self.cap is not None and self.current_frame_idx >= 0:
            original_position = self.current_frame_idx
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
            ret, frame = self.cap.read()
            # Reset position
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
            if ret:
                return frame
        return self.current_frame  # Return current frame as fallback
    
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
        """Start video playback"""
        if self.cap is not None and self.cap.isOpened() and self.current_video_index >= 0:
            # If at the end of the video, go back to beginning
            if self.current_frame_idx >= self.total_frames - 1:
                self.current_frame_idx = 0
                self.seek(0)
            
            # Start or ensure buffer is running
            if not self.video_buffer.worker_thread or not self.video_buffer.worker_thread.isRunning():
                current_video_path = self.videos[self.current_video_index]["path"]
                self.video_buffer.start_buffering(current_video_path, self.current_frame_idx)
            else:
                self.video_buffer.resume()
            
            # Calculate milliseconds per frame using exact video frame rate
            ms_per_frame = 1000 / self.frame_rate  # Don't round to int for more accuracy
            self.target_frame_time = ms_per_frame
            
            # Reset EWMA frame time and timing variables when starting playback
            self.ewma_frame_time = ms_per_frame
            self.frame_time_history = []  # Clear history
            
            # Wait for buffer to be ready before starting playback
            if not self.buffer_ready and not self.video_buffer.is_buffer_ready():
                self.display.setText("Buffering...")
                return
            
            # Start the playback timer with precise timing
            self.timer.start(int(ms_per_frame))
            self._playing = True
            self.play_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaPause))
            
            # Initialize the high-precision frame timing with perf_counter
            self.last_frame_time = time.perf_counter()
        else:
            QMessageBox.warning(None, "Playback Error", "No valid video is loaded")
    
    def pause(self):
        self.timer.stop()
        self._playing = False
        self.play_button.setIcon(self._invert_icon(QStyle.StandardPixmap.SP_MediaPlay))
        
        # Pause the buffer worker as well
        self.video_buffer.pause()
    
    def is_playing(self):
        return self._playing
    
    def get_current_frame(self):
        return self.current_frame
    
    def get_current_frame_idx(self):
        return self.current_frame_idx
    
    def get_total_frames(self):
        return self.total_frames
    
    def seek(self, frame_idx, store_frame_idx=True):
        """
        Seek to a specific frame in the video
        
        Args:
            frame_idx: The frame index to seek to
            store_frame_idx: Whether to update the stored frame_idx in the video info
        """
        if self.cap is None:
            return False
            
        try:
            # Make sure we don't seek past the end of the video or before start
            safe_frame_idx = max(0, min(frame_idx, self.total_frames - 1))
            
            # Update buffer position
            self.video_buffer.seek(safe_frame_idx)
            
            # Update local frame index
            self.current_frame_idx = safe_frame_idx
            
            # Store the frame index in the video info for this specific video
            if store_frame_idx and self.current_video_index >= 0 and self.current_video_index < len(self.videos):
                self.videos[self.current_video_index]["frame_idx"] = safe_frame_idx
            
            # Update UI
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(safe_frame_idx)
            self.position_slider.blockSignals(False)
            
            # For immediate feedback, read directly from the video
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, safe_frame_idx)
            ret, frame = self.cap.read()
            
            if ret:
                self.current_frame = frame
                self.set_image(frame)
                self.update_frame_counter()
                self.frame_changed.emit(safe_frame_idx)
                return True
            else:
                print(f"Failed to read frame at index {safe_frame_idx}")
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
            # Make sure we don't go past the end
            next_frame = min(self.current_frame_idx + 1, self.total_frames - 1)
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
            # Ensure we don't exceed the total frames
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
        self.frame_counter.setText(f"Frame: {self.current_frame_idx} / {self.total_frames-1}")
    
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
    
    def set_controls_enabled(self, enabled):
        """Enable or disable all player controls"""
        # Disable all buttons
        for button in self.findChildren(QPushButton):
            button.setEnabled(enabled)
        
        # Disable all text inputs
        for line_edit in self.findChildren(QLineEdit):
            line_edit.setEnabled(enabled)
        
        # Disable slider and dropdown
        self.position_slider.setEnabled(enabled)
        self.video_selector.setEnabled(enabled)
        
        # If enabling, update video navigation buttons based on current state
        if enabled:
            self.update_video_navigation_buttons()
    
    # Multiple video management methods
    def add_video(self, video_path, display_name=None):
        """Add a video to the list of loaded videos"""
        if not display_name:
            display_name = os.path.basename(video_path)
            
        # Check if the video is already in the list
        for video in self.videos:
            if video["path"] == video_path:
                # If it is, just switch to it
                index = self.videos.index(video)
                self.switch_to_video(index)
                return True
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                QMessageBox.critical(None, "Error", f"Could not open video file: {video_path}")
                return False
                
            # Get video properties - critical for accurate frame rates
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            
            # Handle invalid frame rates from metadata
            if frame_rate <= 0 or frame_rate > 1000:  # Invalid or unrealistic FPS
                # Try to estimate FPS by analyzing the video more accurately
                frame_count = 0
                start_time = time.perf_counter()
                
                # Sample up to 100 frames to get a more accurate estimate
                max_samples = min(100, total_frames)
                for _ in range(max_samples):
                    ret = cap.grab()  # grab() is faster than read() as it doesn't decode
                    if not ret:
                        break
                    frame_count += 1
                    
                # If we grabbed at least 10 frames, calculate FPS
                if frame_count >= 10:
                    elapsed = time.perf_counter() - start_time
                    if elapsed > 0:
                        estimated_fps = frame_count / elapsed
                        # Apply reasonable bounds to the estimated FPS
                        frame_rate = max(min(estimated_fps, 120), 10)
                    else:
                        frame_rate = 30  # Default if timing failed
                else:
                    frame_rate = 30  # Default fallback
                
                # Reset position to start
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                
                print(f"Estimated video frame rate: {frame_rate:.2f} FPS")
            else:
                print(f"Using video frame rate from metadata: {frame_rate:.2f} FPS")
            
            # Read the first frame for preview
            ret, first_frame = cap.read()
            # Reset position to start
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Create video info dictionary with exact frame rate (not rounded)
            video_info = {
                "path": video_path,
                "name": display_name,
                "cap": cap,
                "frame_idx": 0,  # Always start at frame 0 for newly added videos
                "total_frames": total_frames,
                "frame_rate": frame_rate,  # Store exact frame rate
                "current_frame": first_frame.copy() if ret else None
            }
            
            # Add to videos list
            self.videos.append(video_info)
            
            # Update the video selector dropdown
            self.video_selector.addItem(display_name)
            
            # If this is the first video, initialize buffer and select it
            if len(self.videos) == 1:
                self.video_buffer.start_buffering(video_path, 0)
                self.switch_to_video(0)
            else:
                # Otherwise just select the newly added video
                self.video_selector.setCurrentIndex(len(self.videos) - 1)
            
            # Update navigation buttons
            self.update_video_navigation_buttons()
            
            return True
            
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error adding video: {str(e)}")
            return False
    
    def update_video_navigation_buttons(self):
        """Enable/disable video navigation buttons based on current state"""
        has_videos = len(self.videos) > 0
        has_multiple_videos = len(self.videos) > 1
        has_prev = self.current_video_index > 0
        has_next = self.current_video_index < len(self.videos) - 1
        
        # Enable/disable video selector and buttons
        self.video_selector.setEnabled(has_videos)
        self.prev_video_button.setEnabled(has_prev)
        self.next_video_button.setEnabled(has_next)
        
        # Update general controls
        has_current_video = self.current_video_index >= 0 and self.current_video_index < len(self.videos)
        self.position_slider.setEnabled(has_current_video)
        self.play_button.setEnabled(has_current_video)
        self.prev_button.setEnabled(has_current_video)
        self.next_button.setEnabled(has_current_video)
        self.skip_backward.setEnabled(has_current_video)
        self.skip_forward.setEnabled(has_current_video)
        self.goto_button.setEnabled(has_current_video)
    
    def switch_to_video(self, index):
        """Switch to the video at the specified index"""
        if index < 0 or index >= len(self.videos):
            return False
        
        if index == self.current_video_index:
            return True
        
        # Store current video state if there is one
        if self.current_video_index >= 0 and self.current_video_index < len(self.videos):
            current_video = self.videos[self.current_video_index]
            # Make sure to store current frame position, but validate it's in range
            if self.cap and self.cap.isOpened():
                max_frame = max(0, current_video["total_frames"] - 1)
                safe_frame = min(self.current_frame_idx, max_frame)
                current_video["frame_idx"] = safe_frame
        
        was_playing = self._playing
        self.pause()
        
        # Clear buffer and stop worker thread
        self.video_buffer.stop_buffering()
        self.buffer_ready = False
        
        # Update current video info
        self.current_video_index = index
        video_info = self.videos[index]
        self.cap = video_info["cap"]
        self.total_frames = video_info["total_frames"]
        
        # Always use the exact frame rate from the video metadata without rounding
        self.frame_rate = video_info["frame_rate"]
        
        # Make sure the stored frame index is valid for this video
        self.current_frame_idx = min(video_info["frame_idx"], max(0, self.total_frames - 1))
        
        # Update UI elements
        self.position_slider.blockSignals(True)
        self.position_slider.setRange(0, max(0, self.total_frames - 1))
        self.position_slider.setValue(self.current_frame_idx)
        self.position_slider.blockSignals(False)
        
        self.frame_input.setText(str(self.current_frame_idx))
        self.frame_counter.setText(f"Frame: {self.current_frame_idx} / {self.total_frames - 1}")
        
        # Use the exact frame rate for the rate_text field
        self.rate_text.setText(f"{self.frame_rate:.2f}")
        
        if self.video_selector.currentIndex() != index:
            self.video_selector.blockSignals(True)
            self.video_selector.setCurrentIndex(index)
            self.video_selector.blockSignals(False)
        
        self.update_video_info()
        
        # Reset performance metrics for the new video
        self.frame_times = []
        self.dropped_frames = 0
        
        # Start buffering the new video
        self.video_buffer.start_buffering(video_info["path"], self.current_frame_idx)
        
        # Show the initial frame
        try:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                self.set_image(frame)
            else:
                print(f"Warning: Couldn't seek to frame {self.current_frame_idx}, displaying first frame instead")
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = self.cap.read()
                if ret:
                    self.current_frame = frame
                    self.set_image(frame)
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        except Exception as e:
            print(f"Error seeking to frame {self.current_frame_idx}: {str(e)}")
        
        self.update_video_navigation_buttons()
        
        # Restart playback if it was playing
        if was_playing:
            self.play()
        
        self.video_changed.emit(index)
        return True
    
    def prev_video(self):
        """Switch to the previous video in the list"""
        if self.current_video_index > 0:
            self.switch_to_video(self.current_video_index - 1)
    
    def next_video(self):
        """Switch to the next video in the list"""
        if self.current_video_index < len(self.videos) - 1:
            self.switch_to_video(self.current_video_index + 1)
    
    def on_video_selected(self, index):
        """Handle selection of a video from the dropdown"""
        if index >= 0 and index < len(self.videos) and index != self.current_video_index:
            self.switch_to_video(index)
    
    def get_current_video_path(self):
        """Get the path of the currently selected video"""
        if self.current_video_index >= 0 and self.current_video_index < len(self.videos):
            return self.videos[self.current_video_index]["path"]
        return None

    def clear_all_videos(self):
        """Clear all videos from the player"""
        # Pause playback if active
        self.pause()
        
        # Stop buffering
        self.video_buffer.stop_buffering()
        
        # Close all video captures
        for video in self.videos:
            if 'cap' in video and video['cap'] is not None:
                video['cap'].release()
        
        # Reset video player state
        self.videos.clear()
        self.current_video_index = -1
        self.current_frame_idx = 0
        self.total_frames = 0
        self._playing = False
        self.cap = None
        self.current_frame = None
        self.buffer_ready = False
        
        # Clear the video selector dropdown
        self.video_selector.clear()
        
        # Clear the display
        self.display.setText("No video loaded")
        self.info_label.setText("No video loaded")
        self.frame_counter.setText("Frame: 0 / 0")
        self.buffer_status.setValue(0)
        
        # Reset the slider
        self.position_slider.setRange(0, 0)
        self.position_slider.setValue(0)
        
        # Update navigation buttons
        self.update_video_navigation_buttons()
        
        # Signal change
        self.video_changed.emit(-1)

    def update_video_info(self):
        """Update the video info display - separate method for better performance"""
        if self.cap is None:
            return
            
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.video_source_type = self.videos[self.current_video_index]["name"]
        self.video_info = f"Source: {self.video_source_type} | Resolution: {width}x{height} | FPS: {self.frame_rate:.2f} | Frames: {self.total_frames}"
        self.info_label.setText(self.video_info)

    # Remove the update_performance_display method as it's no longer needed
    
    def toggle_high_adaptive_fps(self):
        """Toggle between normal and high adaptive FPS modes"""
        self.high_adaptive_fps_mode = not self.high_adaptive_fps_mode
        
        # Update button text
        self.high_perf_button.setText(f"Adaptive: {'High' if self.high_adaptive_fps_mode else 'Normal'}")
        
        # If turning on high adaptive FPS mode, we might want to increase buffer size
        if self.high_adaptive_fps_mode:
            # Higher buffer for more aggressive timing adjustments
            self.video_buffer.buffer_size = max(40, self.video_buffer.buffer_size)
        else:
            # Reset to default buffer size when returning to normal mode
            self.video_buffer.adjust_buffer_size()
    
    def on_buffer_status_updated(self, percentage):
        """Handle buffer status updates"""
        self.buffer_status.setValue(int(percentage))
    
    def on_buffer_ready(self):
        """Handle buffer ready signal"""
        self.buffer_ready = True
        if self._playing:
            # If we were waiting for buffer, start playback now
            self.play()
    
    def on_end_of_file_reached(self):
        """Handle the single EOF signal - just sets the flag, doesn't stop playback"""
        self.eof_reached = True
        print(f"End of file reached. Buffer has {self.video_buffer.frame_buffer.qsize()} frames remaining.")
        # Don't pause playback here - let nextFrameSlot handle it when buffer is empty
    
    def __del__(self):
        """Destructor to ensure resources are properly released"""
            
        # Stop buffer
        if hasattr(self, 'video_buffer'):
            self.video_buffer.stop_buffering()
            
        # Then release video captures
        if hasattr(self, 'videos'):
            for video in self.videos:
                if 'cap' in video and video['cap'] is not None:
                    video['cap'].release()
        elif hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()