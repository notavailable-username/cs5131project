import cv2
import numpy as np
import logging

# Set up logging
logger = logging.getLogger(__name__)

class MotionDetector:
    def __init__(self, history=500, varThreshold=16):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(history=history, varThreshold=varThreshold)
        self.history = history
        self.varThreshold = varThreshold
        logger.debug(f"Initialized MotionDetector with history={history}, varThreshold={varThreshold}")
    
    def detect(self, frame):
        """Detect moving regions in a frame and return bounding boxes."""
        if frame is None or frame.size == 0:
            logger.error("Invalid frame provided to motion detector")
            return []
            
        try:
            # Apply background subtraction
            mask = self.subtractor.apply(frame)
            
            # Perform morphological operations to reduce noise
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            bboxes = []
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 500:  # filter small regions
                    x, y, w, h = cv2.boundingRect(cnt)
                    bboxes.append((x, y, x+w, y+h))
            
            logger.debug(f"Detected {len(bboxes)} motion regions")
            return bboxes
            
        except Exception as e:
            logger.error(f"Error in motion detection: {e}")
            return []

if __name__ == "__main__":
    import argparse
    import os
    import sys
    
    # Configure logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Test Motion Detection")
    parser.add_argument('--video', type=str, help='Path to video file or camera index (0, 1, etc.)')
    parser.add_argument('--threshold', type=int, default=16, help='varThreshold parameter (default: 16)')
    parser.add_argument('--history', type=int, default=500, help='History length (default: 500)')
    parser.add_argument('--display', action='store_true', help='Display detection results')
    parser.add_argument('--output', type=str, help='Output directory for detection frames')
    args = parser.parse_args()
    
    # Initialize detector
    detector = MotionDetector(history=args.history, varThreshold=args.threshold)
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    # Open video file or camera
    video_source = 0  # Default to camera 0
    if args.video and args.video.isdigit():
        video_source = int(args.video)
    elif args.video:
        video_source = args.video
        if not os.path.exists(video_source):
            print(f"Error: Video file not found: {video_source}")
            sys.exit(1)
    
    print(f"Opening video source: {video_source}")
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {video_source}")
        sys.exit(1)
    
    frame_count = 0
    detection_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            boxes = detector.detect(frame)
            
            if boxes:
                detection_count += len(boxes)
                display_frame = frame.copy()
                
                # Draw boxes
                for box in boxes:
                    x1, y1, x2, y2 = box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Save frame if output directory is specified
                if args.output:
                    output_path = os.path.join(args.output, f"detection_{frame_count:06d}.jpg")
                    cv2.imwrite(output_path, display_frame)
                
                # Display detection
                if args.display:
                    cv2.imshow("Motion Detection", display_frame)
            elif args.display:
                cv2.imshow("Motion Detection", frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Processed {frame_count} frames, detected {detection_count} objects")
            
            if args.display and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Error during processing: {e}")
    finally:
        cap.release()
        if args.display:
            cv2.destroyAllWindows()
        
    print(f"Processing complete. Processed {frame_count} frames with {detection_count} detections.")
