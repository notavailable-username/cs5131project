from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QTabWidget, QPushButton, QFileDialog, QListWidget, QLabel
)
from PyQt6.QtCore import Qt
from .video_player import VideoPlayer
from models.motion_detector import MotionDetector
from models.few_shot import FewShotEnsemble

class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("Motion-Aware Few-Shot Object Detection")
        self.resize(1200, 800)
        self._init_ui()
        # Initialize our models
        md_conf = config.get("motion_detector", {})
        self.motion_detector = MotionDetector(**md_conf)
        fs_conf = config.get("few_shot", {})
        self.few_shot = FewShotEnsemble(num_models=fs_conf.get("models", 3),
                                        confidence_threshold=fs_conf.get("confidence_threshold", 0.5))
    
    def _init_ui(self):
        # Main widget and layout
        central_widget = QWidget()
        main_layout = QHBoxLayout()
        
        # Left: Video/Image display
        self.video_player = VideoPlayer()
        self.video_player.setFixedSize(640, 480)
        main_layout.addWidget(self.video_player)
        
        # Right: Tabs and controls
        self.tabs = QTabWidget()
        self.tabs.addTab(self._create_import_tab(), "Import")
        self.tabs.addTab(self._create_annotation_tab(), "Annotation")
        self.tabs.addTab(self._create_difficulty_tab(), "Difficult Images")
        
        main_layout.addWidget(self.tabs)
        
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
    
    def _create_import_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        btn_load_image = QPushButton("Load Image")
        btn_load_image.clicked.connect(self.load_image)
        layout.addWidget(btn_load_image)
        
        btn_load_video = QPushButton("Load Video")
        btn_load_video.clicked.connect(self.load_video)
        layout.addWidget(btn_load_video)
        
        widget.setLayout(layout)
        return widget
    
    def _create_annotation_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        
        btn_detect = QPushButton("Run Motion Detection & Few-Shot Annotation")
        btn_detect.clicked.connect(self.run_annotation)
        layout.addWidget(btn_detect)
        
        self.annotation_status = QLabel("Status: Idle")
        layout.addWidget(self.annotation_status)
        
        widget.setLayout(layout)
        return widget
    
    def _create_difficulty_tab(self):
        widget = QWidget()
        layout = QVBoxLayout()
        self.difficult_list = QListWidget()
        layout.addWidget(QLabel("Images with Low Confidence:"))
        layout.addWidget(self.difficult_list)
        
        btn_manual = QPushButton("Manual Annotate Selected")
        btn_manual.clicked.connect(self.manual_annotate)
        layout.addWidget(btn_manual)
        
        widget.setLayout(layout)
        return widget
    
    def load_image(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Image Files (*.png *.jpg *.jpeg)")
        if file_name:
            self.video_player.load_image(file_name)
    
    def load_video(self):
        file_name, _ = QFileDialog.getOpenFileName(self, "Select Video", "", "Video Files (*.mp4 *.avi)")
        if file_name:
            self.video_player.load_video(file_name)
    
    def run_annotation(self):
        # For the current frame (or a loaded image), run motion detection and few-shot annotation.
        # In a full implementation, you would iterate over all frames/images.
        frame = self._get_current_frame()
        if frame is None:
            self.annotation_status.setText("No frame loaded.")
            return
        
        boxes = self.motion_detector.detect(frame)
        self.annotation_status.setText(f"Detected {len(boxes)} moving objects.")
        
        # For each box, crop image patch and run few-shot prediction
        difficult_images = []
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            patch = frame[y1:y2, x1:x2]
            label, conf = self.few_shot.predict(patch)
            # Here you would draw the bounding box and label on the frame
            # For low-confidence predictions, add to difficult list.
            if conf < self.config.get("few_shot", {}).get("confidence_threshold", 0.5):
                difficult_images.append(f"Box {bbox}: {label} ({conf:.2f})")
        self.difficult_list.clear()
        self.difficult_list.addItems(difficult_images)
    
    def _get_current_frame(self):
        # Retrieve current frame from video_player if available.
        # For simplicity, we assume that if a video is playing, we capture the current pixmap.
        pixmap = self.video_player.pixmap()
        if pixmap is None:
            return None
        # Convert QPixmap back to cv2 image (this is a placeholder; in practice, maintain frame data)
        # For now, we load the image from file if available.
        return None
    
    def manual_annotate(self):
        # Open a new window or dialog for manual annotation.
        # This is a placeholder – you would implement annotation tools (drawing boxes, selecting classes).
        print("Manual annotation triggered.")

