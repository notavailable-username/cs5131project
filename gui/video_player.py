import cv2
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel, QMessageBox

class VideoPlayer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrameSlot)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.current_frame = None
        self._playing = False
        self.setMinimumSize(640, 480)
        self.frame_rate = 30  # Default frame rate
    
    def load_video(self, video_path):
        try:
            self.cap = cv2.VideoCapture(video_path)
            if not self.cap.isOpened():
                QMessageBox.critical(None, "Error", f"Could not open video file: {video_path}")
                return False
                
            self.current_frame_idx = 0
            ret, frame = self.cap.read()
            if ret:
                self.frame_rate = self.cap.get(cv2.CAP_PROP_FPS)
                if self.frame_rate <= 0:
                    self.frame_rate = 30  # Fallback frame rate
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
        self.cap = None
        image = cv2.imread(image_path)
        self.set_image(image)
        self.current_frame = image
    
    def nextFrameSlot(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.set_image(frame)
                self.current_frame = frame
                self.current_frame_idx += 1
            else:
                self.timer.stop()
                self._playing = False
    
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
            self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, 
                                       Qt.TransformationMode.SmoothTransformation))
        except Exception as e:
            print(f"Error displaying frame: {str(e)}")
    
    def set_frame(self, frame):
        self.current_frame = frame
        self.set_image(frame)
    
    def play(self):
        if self.cap is not None and self.cap.isOpened():
            # Calculate milliseconds per frame for smoother playback
            ms_per_frame = int(1000 / self.frame_rate)
            self.timer.start(ms_per_frame)
            self._playing = True
        else:
            QMessageBox.warning(None, "Playback Error", "No valid video is loaded")
    
    def pause(self):
        self.timer.stop()
        self._playing = False
    
    def is_playing(self):
        return self._playing
    
    def get_current_frame(self):
        return self.current_frame
    
    def seek(self, frame_idx):
        if self.cap is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = self.cap.read()
            if ret:
                self.set_image(frame)
                self.current_frame = frame
                self.current_frame_idx = frame_idx
    
    def resizeEvent(self, event):
        if self.current_frame is not None:
            self.set_image(self.current_frame)
        super().resizeEvent(event)
