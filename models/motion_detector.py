import cv2
import numpy as np

class MotionDetector:
    def __init__(self, history=500, varThreshold=16):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold)
    
    def detect(self, frame):
        """Detect moving regions in a frame and return bounding boxes."""
        mask = self.subtractor.apply(frame)
        # Perform morphological operations to reduce noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bboxes = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 500:  # filter small regions
                x, y, w, h = cv2.boundingRect(cnt)
                bboxes.append((x, y, x+w, y+h))
        return bboxes

if __name__ == "__main__":
    # Quick test on a video file or camera
    cap = cv2.VideoCapture(0)
    detector = MotionDetector()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        boxes = detector.detect(frame)
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imshow("Motion Detection", frame)
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
