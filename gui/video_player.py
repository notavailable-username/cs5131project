import cv2
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QLabel

class VideoPlayer(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.cap = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.nextFrameSlot)
    
    def load_video(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        self.timer.start(30)
    
    def load_image(self, image_path):
        self.cap = None
        image = cv2.imread(image_path)
        self.set_image(image)
    
    def nextFrameSlot(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.set_image(frame)
            else:
                self.timer.stop()
    
    def set_image(self, frame):
        # Convert BGR to RGB and set image in QLabel
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        self.setPixmap(pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
