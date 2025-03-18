from PyQt6.QtCore import QObject, QThread, pyqtSignal, QMutex, QWaitCondition, QTimer
import cv2
import time
import queue
import numpy as np
import psutil
import os

class FrameBufferWorker(QThread):
    """Worker thread for decoding frames in the background"""
    
    buffer_status_changed = pyqtSignal(float)  # Signal to report buffer fill percentage
    buffer_ready = pyqtSignal()  # Signal when buffer is ready for playback
    frame_decoded = pyqtSignal(int)  # Signal when a frame is decoded (frame_idx)
    end_of_file_reached = pyqtSignal()  # Single signal for when EOF is reached
    
    def __init__(self, video_path, buffer, start_frame=0, buffer_size=30, parent=None):
        """Initialize the worker thread
        
        Args:
            video_path: Path to the video file
            buffer: The buffer queue to fill with frames
            start_frame: Frame index to start buffering from
            buffer_size: Maximum number of frames to buffer
            parent: Parent QObject
        """
        super().__init__(parent)
        self.video_path = video_path
        self.buffer = buffer
        self.current_frame_idx = start_frame
        self.max_buffer_size = buffer_size
        
        self.cap = None
        self.running = False
        self.paused = False
        self.seeking = False
        self.seek_position = -1
        
        self.mutex = QMutex()
        self.condition = QWaitCondition()
        
        # Single flag to track if we've reached end of file
        self.eof_reached = False

    def run(self):
        """Main worker thread loop for frame decoding"""
        self.running = True
        self.cap = cv2.VideoCapture(self.video_path)
        self.eof_reached = False  # Reset EOF flag
        
        if not self.cap.isOpened():
            print(f"Error: Could not open video: {self.video_path}")
            return
            
        # Set initial position
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        while self.running:
            self.mutex.lock()
            
            # Handle pause state
            if self.paused:
                self.condition.wait(self.mutex)
                
            # Handle seek request
            if self.seeking:
                # Clear buffer when seeking
                while not self.buffer.empty():
                    try:
                        self.buffer.get_nowait()
                    except queue.Empty:
                        break
                
                # Make sure seek position doesn't exceed total frames
                self.seek_position = max(0, min(self.seek_position, total_frames - 1))
                
                # Set new position
                self.current_frame_idx = self.seek_position
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
                self.seeking = False
                self.eof_reached = False   # Reset EOF flag
                
            self.mutex.unlock()
            
            # If buffer is full, wait
            if self.buffer.qsize() >= self.max_buffer_size:
                # Buffer is full, emit status and sleep briefly
                self.buffer_status_changed.emit(100.0)
                time.sleep(0.01)  # Short sleep to prevent CPU overuse
                continue
            
            # If we've reached EOF, sleep to avoid busy loop
            if self.eof_reached:
                time.sleep(0.1)  # Sleep briefly to avoid busy loop
                continue
                
            # Read the next frame - THIS IS THE ONLY PLACE WHERE EOF IS DETECTED
            ret, frame = self.cap.read()
            
            if not ret:
                # We've reached the end of the file - SINGLE POINT OF EOF DETECTION
                if not self.eof_reached:
                    self.eof_reached = True
                    self.end_of_file_reached.emit()  # Emit EOF signal ONCE
                    print(f"End of file reached at position {self.current_frame_idx}")
                
                time.sleep(0.1)  # Sleep briefly 
                continue
                
            # Add frame to buffer with its index
            try:
                frame_copy = frame.copy()  # Make a copy to prevent reference issues
                self.buffer.put((self.current_frame_idx, frame_copy))
                self.frame_decoded.emit(self.current_frame_idx)
                
                # Update buffer status
                fill_percentage = int((self.buffer.qsize() / self.max_buffer_size) * 100)
                self.buffer_status_changed.emit(fill_percentage)
                
                # Signal buffer is ready for initial playback
                frames_left = total_frames - self.current_frame_idx - 1
                if fill_percentage >= 30 or (frames_left < self.max_buffer_size and self.buffer.qsize() > 0):
                    self.buffer_ready.emit()
                    
            except Exception as e:
                print(f"Error adding frame to buffer: {str(e)}")
            
            # Move to next frame
            self.current_frame_idx += 1
            
        # Clean up
        if self.cap:
            self.cap.release()
    
    def seek(self, frame_idx):
        """Request seeking to a specific frame"""
        self.mutex.lock()
        self.seeking = True
        self.seek_position = frame_idx
        self.paused = False
        self.eof_reached = False   # Reset EOF flag
        self.condition.wakeAll()
        self.mutex.unlock()
            
    def pause(self):
        """Pause the worker thread"""
        self.mutex.lock()
        self.paused = True
        self.mutex.unlock()
        
    def resume(self):
        """Resume the worker thread"""
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        
    def stop(self):
        """Stop the worker thread"""
        self.running = False
        self.mutex.lock()
        self.paused = False
        self.condition.wakeAll()
        self.mutex.unlock()
        self.wait()


class VideoBuffer(QObject):
    """A thread-safe video frame buffer with prefetching capabilities"""
    
    buffer_status_updated = pyqtSignal(float)  # Buffer fill percentage
    buffer_ready = pyqtSignal()  # Buffer is ready for playback
    end_of_file_reached = pyqtSignal()  # Single signal for EOF notification
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.frame_buffer = queue.Queue()
        self.worker_thread = None
        self.buffer_size = 30  # Default buffer size
        self.current_video_path = None
        
        # Performance monitoring with high precision timer
        self.decode_times = []
        self.dropped_frames = 0
        self.last_performance_update = time.perf_counter()
        self.performance_update_interval = 0.5  # 500ms
        
        # Adaptive buffer size
        self.auto_adjust_buffer = True
        self.system_memory = psutil.virtual_memory().total / (1024*1024*1024)  # GB
        self.adjust_buffer_size()
        
        # Add a separate timer for regular updates
        self.update_timer = QTimer(self)
        self.update_timer.timeout.connect(self._update_performance_stats)
        self.update_timer.start(500)  # 500ms update interval
        self.eof_reached = False  # Single EOF tracking flag

    def adjust_buffer_size(self, frame_rate=None):
        """Adaptively adjust buffer size based on system memory and frame rate"""
        old_buffer_size = self.buffer_size
        
        if self.auto_adjust_buffer:
            # If frame rate is provided, set buffer to exactly 2x the frame rate
            if frame_rate is not None and frame_rate > 0:
                # Use exactly 2x frame rate as specified
                frame_rate_based_size = int(frame_rate * 2)  # Changed from 5x to 2x
                self.buffer_size = max(10, frame_rate_based_size)  # Min 10 frames but no upper cap
            else:
                # Fall back to memory-based calculation only if frame rate not provided
                mem_based_size = int(self.system_memory * 100)  # 100 frames per GB of RAM
                self.buffer_size = max(10, mem_based_size)
        
        # Print detailed log message showing the calculation
        print(f"Buffer size adjusted from {old_buffer_size} to {self.buffer_size} frames")
        if frame_rate is not None:
            print(f"Video frame rate: {frame_rate:.2f} FPS (2x = {int(frame_rate * 2)} frames)")  # Changed from 5x to 2x

    def start_buffering(self, video_path, start_frame=0):
        """Start the buffer worker for a video"""
        self.stop_buffering()  # Clean up any existing worker
        
        self.current_video_path = video_path
        
        # Clear any existing frames
        while not self.frame_buffer.empty():
            try:
                self.frame_buffer.get_nowait()
            except queue.Empty:
                break
        
        # Get the video frame rate to adjust buffer size
        try:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                frame_rate = cap.get(cv2.CAP_PROP_FPS)
                # Also get total frames for accurate frame positioning
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                print(f"Loading video: {os.path.basename(video_path)}")
                print(f"Total frames: {total_frames}, Frame rate: {frame_rate:.2f} FPS")
                cap.release()
                
                # Adjust buffer size based on frame rate - ensure this happens for every video
                if frame_rate > 0:
                    self.adjust_buffer_size(frame_rate)
                else:
                    print(f"Warning: Invalid frame rate ({frame_rate}), using default buffer size")
            else:
                print(f"Warning: Could not open video to get frame rate")
        except Exception as e:
            print(f"Error reading video properties: {e}")
                
        # Create and start worker thread
        self.worker_thread = FrameBufferWorker(
            video_path, 
            self.frame_buffer, 
            start_frame, 
            self.buffer_size
        )
        
        # Reset EOF state when starting new buffering
        self.eof_reached = False
        
        # Connect signals
        self.worker_thread.buffer_status_changed.connect(self.on_buffer_status_changed)
        self.worker_thread.buffer_ready.connect(self.on_buffer_ready)
        self.worker_thread.frame_decoded.connect(self.on_frame_decoded)
        self.worker_thread.end_of_file_reached.connect(self.on_end_of_file_reached)
        
        # Start the worker
        self.worker_thread.start()
        
    def stop_buffering(self):
        """Stop the buffer worker and clean up"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()
            self.worker_thread = None
            
    def seek(self, frame_idx):
        """Request seek to a specific frame"""
        if self.worker_thread and self.worker_thread.isRunning():
            # Keep track that we've initiated a seek operation
            self.worker_thread.seek(frame_idx)
            
    def pause(self):
        """Pause the buffer worker"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.pause()
            
    def resume(self):
        """Resume the buffer worker"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.resume()
            
    def get_frame(self, expected_frame_idx=None):
        """Get the next frame from the buffer - NO signals emitted here"""
        start_time = time.perf_counter()
        
        try:
            frame_idx, frame = self.frame_buffer.get(block=False)
            
            # Record decode time for performance monitoring
            decode_time = time.perf_counter() - start_time
            self.decode_times.append(decode_time)
            
            # Keep only recent measurements for performance stats
            if len(self.decode_times) > 100:
                self.decode_times = self.decode_times[-100:]
                
            # Still count dropped frames but don't generate warning
            if expected_frame_idx is not None and frame_idx != expected_frame_idx:
                self.dropped_frames += 1
            
            return frame_idx, frame
        except queue.Empty:
            # Return None, None for empty buffer - no signals emitted here
            return None, None
            
    def get_buffer_level(self):
        """Get the current buffer fill percentage"""
        if self.buffer_size > 0:
            return (self.frame_buffer.qsize() / self.buffer_size) * 100
        return 0
        
    def is_buffer_ready(self):
        """Check if buffer is sufficiently filled to begin playback"""
        # Buffer is ready if it's at least 30% full
        buffer_size = self.frame_buffer.qsize()
        
        # Check if worker exists and if we're near the end of the video
        near_end = False
        if self.worker_thread and self.worker_thread.cap:
            total_frames = int(self.worker_thread.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            current_idx = self.worker_thread.current_frame_idx
            frames_left = total_frames - current_idx
            near_end = frames_left < self.buffer_size
        
        # Buffer is ready if:
        # 1. It has sufficient frames (at least 30% full)
        # 2. OR we're near the end of the video with at least 1 frame buffered
        return (buffer_size / self.buffer_size) >= 0.3 or (near_end and buffer_size > 0)
        
    def _update_performance_stats(self):
        """Update performance statistics regularly using dedicated timer"""
        stats = {
            'buffer_level': self.get_buffer_level(),
            'buffer_size': self.buffer_size,
            'buffer_frames': self.frame_buffer.qsize(),
            'dropped_frames': self.dropped_frames,
            'avg_decode_time': sum(self.decode_times) / max(1, len(self.decode_times)) if self.decode_times else 0
        }
        
        self.last_performance_update = time.perf_counter()
    
    def on_buffer_status_changed(self, percentage):
        """Handle buffer status change from worker thread"""
        # To avoid glitching between 99% and 100%, we'll only emit changes if they're significant
        # or at the extremes (0% or 100%)
        if (percentage == 0 or percentage == 100 or 
            not hasattr(self, '_last_emitted_percentage') or 
            abs(self._last_emitted_percentage - percentage) >= 5):
            
            self._last_emitted_percentage = percentage
            self.buffer_status_updated.emit(percentage)
        
    def on_buffer_ready(self):
        """Handle buffer ready signal from worker thread"""
        self.buffer_ready.emit()
        
    def on_frame_decoded(self, frame_idx):
        """Handle frame decoded signal from worker thread"""
        # This could be used for more detailed tracking if needed
        pass
    
    def on_video_ended(self):
        """Handle video ended signal from worker thread"""
        # Forward the signal to the player
        self.video_ended.emit()
    
    def on_end_of_file_reached(self):
        """Handle end of file reached signal from worker thread"""
        # Set flag and forward signal - THIS IS THE ONLY PLACE EOD SIGNAL IS FORWARDED
        self.eof_reached = True
        
        # Update buffer status to 100% automatically when EOF is reached
        self.buffer_status_updated.emit(100.0)
        self._last_emitted_percentage = 100.0  # Update last emitted percentage to avoid immediate overwrite
        
        # Forward the EOF signal to VideoPlayer
        self.end_of_file_reached.emit()
        
        print(f"End of file reached. Buffer has {self.frame_buffer.qsize()} frames remaining.")

    def __del__(self):
        """Clean up resources when object is destroyed"""
            
        self.stop_buffering()
