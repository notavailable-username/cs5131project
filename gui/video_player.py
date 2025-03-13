import cv2
from PyQt6.QtCore import QTimer, Qt, pyqtSignal
from PyQt6.QtGui import QImage, QPixmap, QIcon
from PyQt6.QtWidgets import (QLabel, QMessageBox, QWidget, QVBoxLayout, 
                            QHBoxLayout, QPushButton, QSlider, QSpinBox, QStyle, QSizePolicy, QFrame, QLineEdit)

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
        
        self.setupUI()
    
    def setupUI(self):
        # Main layout
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
        
        # Buttons layout - First row
        buttons_layout = QHBoxLayout()
        
        # Play/Pause button with icon
        self.play_button = QPushButton()
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.clicked.connect(self.toggle_playback)
        self.play_button.setToolTip("Play/Pause")
        buttons_layout.addWidget(self.play_button)
        
        # Previous frame
        self.prev_button = QPushButton()
        self.prev_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward))
        self.prev_button.clicked.connect(self.prev_frame)
        self.prev_button.setToolTip("Previous Frame")
        buttons_layout.addWidget(self.prev_button)
        
        # Next frame
        self.next_button = QPushButton()
        self.next_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward))
        self.next_button.clicked.connect(self.next_frame)
        self.next_button.setToolTip("Next Frame")
        buttons_layout.addWidget(self.next_button)
        
        # Add first row of buttons to layout
        control_layout.addLayout(buttons_layout)
        
        # Skip buttons layout - Second row
        skip_buttons_layout = QHBoxLayout()
        
        # Frame skip amount
        self.skip_spinbox = QSpinBox()
        self.skip_spinbox.setRange(1, 100)
        self.skip_spinbox.setValue(self.skip_frames)
        self.skip_spinbox.valueChanged.connect(self.update_skip_amount)
        self.skip_spinbox.setToolTip("Number of frames to skip")
        skip_buttons_layout.addWidget(self.skip_spinbox)

        # Skip backward button
        self.skip_backward = QPushButton()
        self.skip_backward.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward))
        self.skip_backward.clicked.connect(self.skip_backward_frames)
        self.skip_backward.setToolTip("Skip Backward")
        skip_buttons_layout.addWidget(self.skip_backward)
        
        # Skip forward button
        self.skip_forward = QPushButton()
        self.skip_forward.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward))
        self.skip_forward.clicked.connect(self.skip_forward_frames)
        self.skip_forward.setToolTip("Skip Forward")
        skip_buttons_layout.addWidget(self.skip_forward)        
        
        # Add second row of buttons to layout
        control_layout.addLayout(skip_buttons_layout)
        
        # Frame navigation layout - Third row
        frame_nav_layout = QHBoxLayout()
        
        # Add go to specific frame functionality
        self.frame_input = QSpinBox()
        self.frame_input.setRange(0, 0)
        self.frame_input.setToolTip("Enter frame number")
        frame_nav_layout.addWidget(self.frame_input)
        
        self.goto_button = QPushButton("Go to Frame")
        self.goto_button.clicked.connect(self.goto_frame)
        self.goto_button.setToolTip("Jump to specified frame")
        frame_nav_layout.addWidget(self.goto_button)
        
        # Add third row to layout
        control_layout.addLayout(frame_nav_layout)
        
        # Video information display
        self.info_label = QLabel("No video loaded")
        control_layout.addWidget(self.info_label)
        
        main_layout.addLayout(control_layout)
        
        self.setLayout(main_layout)
    
    def load_video(self, video_path):
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
            self.frame_input.setRange(0, self.total_frames - 1)
            self.frame_counter.setText(f"Frame: 0 / {self.total_frames}")
            
            # Update video info
            width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_info = f"Resolution: {width}x{height} | FPS: {self.frame_rate:.2f} | Frames: {self.total_frames}"
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
    
    def load_image(self, image_path):
        try:
            frame = cv2.imread(image_path)
            if frame is None:
                QMessageBox.critical(None, "Error", f"Could not open image file: {image_path}")
                return False
                
            self.current_frame = frame
            self.set_image(frame)
            self.info_label.setText(f"Image loaded: {image_path}")
            return True
        except Exception as e:
            QMessageBox.critical(None, "Error", f"Error loading image: {str(e)}")
            return False
    
    def nextFrameSlot(self):
        if self.cap is None or not self.cap.isOpened():
            return
            
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
    
    def play(self):
        if self.cap is not None and self.cap.isOpened():
            # Calculate milliseconds per frame for smoother playback
            ms_per_frame = int(1000 / self.frame_rate)
            self.timer.start(ms_per_frame)
            self._playing = True
            self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause))
        else:
            QMessageBox.warning(None, "Playback Error", "No valid video is loaded")
    
    def pause(self):
        self.timer.stop()
        self._playing = False
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
    
    def is_playing(self):
        return self._playing
    
    def get_current_frame(self):
        return self.current_frame
    
    def get_current_frame_idx(self):
        return self.current_frame_idx
    
    def get_total_frames(self):
        return self.total_frames
    
    def seek(self, frame_idx):
        if self.cap is not None:
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
    
    def update_skip_amount(self, value):
        """Update the number of frames to skip"""
        self.skip_frames = value
    
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
            frame_num = self.frame_input.value()
            if self.cap is not None:
                self.pause()
                self.seek(frame_num)
        except ValueError:
            QMessageBox.warning(None, "Invalid Input", "Please enter a valid frame number")
