import os
import shutil
import json
import cv2
import yaml
import logging
import random
import ctypes
from datetime import datetime
from enum import Enum
from typing import Dict, List, Tuple, Optional, Union

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LabelSource(Enum):
    HUMAN = "human"
    AUTO = "auto"

class LabelType(Enum):
    BOUNDING_BOX = "bounding_box"  # Only coordinates
    OBJECT_DETECTION = "object_detection"  # Coordinates + classification

class DatasetManager:
    """
    Manages dataset directory structure and file operations
    """
    def __init__(self, base_dir: str = None):
        """
        Initialize the dataset manager
        
        Args:
            base_dir: Base directory for datasets. If None, uses 'datasets' in project dir
        """
        if base_dir is None:
            # Default to datasets folder in project directory
            project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            self.base_dir = os.path.join(project_dir, 'datasets')
        else:
            self.base_dir = base_dir
        
        # Define directory structure
        self.dirs = {
            'videos': {
                'main': os.path.join(self.base_dir, 'videos'),
                'processed': os.path.join(self.base_dir, 'videos', 'processed'),
                'unprocessed': os.path.join(self.base_dir, 'videos', 'unprocessed'),
            },
            'images': {
                'main': os.path.join(self.base_dir, 'images'),
                'unlabeled': os.path.join(self.base_dir, 'images', 'unlabeled'),
                'labeled': os.path.join(self.base_dir, 'images', 'labeled'),
            },
            'labels': {
                'main': os.path.join(self.base_dir, 'labels'),
                'bounding_boxes': {
                    'main': os.path.join(self.base_dir, 'labels', 'bounding_boxes'),
                    'human': os.path.join(self.base_dir, 'labels', 'bounding_boxes', 'human'),
                    'auto': os.path.join(self.base_dir, 'labels', 'bounding_boxes', 'auto'),
                },
                'object_detection': {
                    'main': os.path.join(self.base_dir, 'labels', 'object_detection'),
                    'human': os.path.join(self.base_dir, 'labels', 'object_detection', 'human'),
                    'auto': os.path.join(self.base_dir, 'labels', 'object_detection', 'auto'),
                }
            },
            'few_shot_examples': os.path.join(self.base_dir, 'few_shot_examples'),
            'yolo': {
                'main': os.path.join(self.base_dir, 'yolo'),
                'images': {
                    'train': os.path.join(self.base_dir, 'yolo', 'images', 'train'),
                    'val': os.path.join(self.base_dir, 'yolo', 'images', 'val'),
                },
                'labels': {
                    'train': os.path.join(self.base_dir, 'yolo', 'labels', 'train'),
                    'val': os.path.join(self.base_dir, 'yolo', 'labels', 'val'),
                }
            },
        }
        self._metadata_file = os.path.join(self.base_dir, 'dataset_metadata.json')
        
    def create_directory_structure(self) -> None:
        """Create the full directory structure if it doesn't exist"""
        logger.info(f"Creating dataset directory structure at {self.base_dir}")
        
        # Recursively create all directories
        def create_dirs(dir_dict):
            for key, value in dir_dict.items():
                if isinstance(value, dict):
                    create_dirs(value)
                else:
                    os.makedirs(value, exist_ok=True)
                    
        create_dirs(self.dirs)
        logger.info("Directory structure created successfully")
        
        # Initialize metadata file if it doesn't exist
        if not os.path.exists(self._metadata_file):
            self._save_metadata({'videos': {}, 'images': {}, 'labels': {}})
    
    def _load_metadata(self) -> Dict:
        """Load dataset metadata from file"""
        if os.path.exists(self._metadata_file):
            try:
                with open(self._metadata_file, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                logger.error(f"Error decoding metadata file: {self._metadata_file}")
                return {'videos': {}, 'images': {}, 'labels': {}}
            except IOError as e:
                logger.error(f"IO error reading metadata file: {e}")
                return {'videos': {}, 'images': {}, 'labels': {}}
        else:
            return {'videos': {}, 'images': {}, 'labels': {}}
    
    def _save_metadata(self, metadata: Dict) -> None:
        """Save dataset metadata to file"""
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self._metadata_file), exist_ok=True)
            with open(self._metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
        except IOError as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _check_disk_space(self, required_mb: int, path: str) -> bool:
        """Check if there is enough disk space for an operation"""
        try:
            free_bytes = ctypes.c_ulonglong(0)
            ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                ctypes.c_wchar_p(path), None, None, ctypes.pointer(free_bytes))
            free_mb = free_bytes.value // (1024 * 1024)
            
            logger.debug(f"Available disk space: {free_mb} MB, Required: {required_mb} MB")
            return free_mb > required_mb
        except Exception as e:
            logger.error(f"Failed to check disk space: {e}")
            return True  # Assume there's enough space if check fails
    
    def add_video(self, video_path: str) -> str:
        """
        Copy a video to the unprocessed videos directory
        
        Args:
            video_path: Path to the video file
            
        Returns:
            New path of the video in the dataset
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Check file type
        _, video_ext = os.path.splitext(video_path)
        if video_ext.lower() not in ['.mp4', '.avi', '.mov']:
            raise ValueError(f"Unsupported video format: {video_ext}")
        
        # Check disk space
        try:
            file_size_bytes = os.path.getsize(video_path)
            file_size_mb = file_size_bytes // (1024 * 1024)
            
            if not self._check_disk_space(file_size_mb * 2, self.base_dir):  # Double for safety
                raise IOError(f"Not enough disk space to add video (needs ~{file_size_mb*2} MB)")
            
            # Copy to unprocessed folder
            os.makedirs(self.dirs['videos']['unprocessed'], exist_ok=True)
            
            video_name = os.path.basename(video_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_name = f"{timestamp}_{video_name}"
            new_path = os.path.join(self.dirs['videos']['unprocessed'], new_name)
            
            shutil.copy2(video_path, new_path)
            logger.info(f"Video added: {video_path} -> {new_path}")
            
            # Update metadata
            metadata = self._load_metadata()
            metadata['videos'][new_name] = {
                'original_path': video_path,
                'status': 'unprocessed',
                'added_date': datetime.now().isoformat(),
            }
            self._save_metadata(metadata)
            
            return new_path
            
        except (IOError, shutil.Error) as e:
            logger.error(f"Failed to copy video: {e}")
            raise
        except OSError as e:
            logger.error(f"OS error during video addition: {e}")
            raise
    
    def mark_video_processed(self, video_name: str) -> str:
        """
        Mark a video as processed by moving it to the processed folder
        
        Args:
            video_name: Name of the video file
            
        Returns:
            New path of the video in the processed folder
        """
        src_path = os.path.join(self.dirs['videos']['unprocessed'], video_name)
        dst_path = os.path.join(self.dirs['videos']['processed'], video_name)
        
        if not os.path.exists(src_path):
            logger.warning(f"Video not found in unprocessed directory: {video_name}")
            return None
        
        try:
            # Move video to processed folder
            os.makedirs(self.dirs['videos']['processed'], exist_ok=True)
            shutil.move(src_path, dst_path)
            logger.info(f"Video marked as processed: {video_name}")
            
            # Update metadata
            metadata = self._load_metadata()
            if video_name in metadata['videos']:
                metadata['videos'][video_name]['status'] = 'processed'
                metadata['videos'][video_name]['processed_date'] = datetime.now().isoformat()
                self._save_metadata(metadata)
                
            return dst_path
            
        except (IOError, shutil.Error) as e:
            logger.error(f"Failed to mark video as processed: {e}")
            return None
    
    def extract_frames(self, video_path: str, interval: int = 30) -> List[str]:
        """ 
        Extract frames from a video and save to images/unlabeled
        
        Args:
            video_path: Path to the video file
            interval: Frame interval to extract (1 = every frame)
            
        Returns:
            List of paths to extracted frames
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        try:
            # Create directory for video frames
            video_name = os.path.basename(video_path)
            video_name_no_ext = os.path.splitext(video_name)[0]
            frames_dir = os.path.join(self.dirs['images']['unlabeled'], video_name_no_ext)
            os.makedirs(frames_dir, exist_ok=True)
            
            # Extract frames
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise IOError(f"Could not open video: {video_path}")
            
            # Calculate disk space needed
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            # Estimate disk space needed (~3 bytes per pixel for JPEG)
            frames_to_extract = frame_count // interval + 1
            required_space_mb = (width * height * 3 * frames_to_extract) / (1024 * 1024)
            required_space_mb *= 1.2  # Add 20% margin
            
            if not self._check_disk_space(required_space_mb, self.base_dir):
                raise IOError(f"Not enough disk space to extract frames (needs ~{required_space_mb} MB)")
            
            frame_count = 0
            extracted_paths = []
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % interval == 0:
                    frame_path = os.path.join(frames_dir, f"frame_{frame_count:06d}.jpg")
                    try:
                        cv2.imwrite(frame_path, frame)
                        extracted_paths.append(frame_path)
                        
                        # Update metadata
                        metadata = self._load_metadata()
                        frame_name = os.path.basename(frame_path)
                        metadata['images'][frame_name] = {
                            'source_video': video_name,
                            'frame_number': frame_count,
                            'status': 'unlabeled',
                            'extracted_date': datetime.now().isoformat(),
                        }
                        self._save_metadata(metadata)
                    except Exception as e:
                        logger.error(f"Failed to save frame {frame_count}: {e}")
                        
                frame_count += 1
            
            cap.release()
            logger.info(f"Extracted {len(extracted_paths)} frames from {video_name}")
            return extracted_paths
            
        except Exception as e:
            logger.error(f"Error extracting frames: {e}")
            if 'cap' in locals() and cap is not None:
                cap.release()
            return []
    
    def save_labels(self, image_name: str, labels: List[Tuple], 
                    label_type: LabelType, source: LabelSource) -> str:
        """
        Save labels for an image
        
        Args:
            image_name: Name of the image file (without path)
            labels: List of labels in YOLO format (class_id, x, y, w, h) or (x1, y1, x2, y2) for bbox only
            label_type: Type of label (BOUNDING_BOX or OBJECT_DETECTION)
            source: Source of the labels (HUMAN or AUTO)
            
        Returns:
            Path to the saved label file
        """
        # Determine the label directory based on type and source
        if label_type == LabelType.BOUNDING_BOX:
            label_dir = self.dirs['labels']['bounding_boxes'][source.value]
        else:
            label_dir = self.dirs['labels']['object_detection'][source.value]
        
        # Create label directory if needed
        os.makedirs(label_dir, exist_ok=True)
        
        # Create label file path
        image_base = os.path.splitext(image_name)[0]
        label_path = os.path.join(label_dir, f"{image_base}.txt")
        
        # Write labels to file
        with open(label_path, 'w') as f:
            for label in labels:
                if label_type == LabelType.BOUNDING_BOX:
                    # x1, y1, x2, y2 format for bounding boxes
                    x1, y1, x2, y2 = label
                    f.write(f"{x1} {y1} {x2} {y2}\n")
                else:
                    # YOLO format for object detection: class_id x_center y_center width height
                    class_id, x_center, y_center, width, height = label
                    f.write(f"{int(class_id)} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")
        
        # Update metadata
        metadata = self._load_metadata()
        metadata['labels'][image_base] = {
            'image': image_name,
            'label_type': label_type.value,
            'source': source.value,
            'label_path': label_path,
            'created_date': datetime.now().isoformat(),
        }
        self._save_metadata(metadata)
        
        logger.info(f"Saved {len(labels)} labels for {image_name} ({source.value}, {label_type.value})")
        return label_path
    
    def load_labels(self, image_name: str, label_type: LabelType = None, 
                   source: LabelSource = None) -> List[Tuple]:
        """
        Load labels for an image
        
        Args:
            image_name: Name of the image file (without path)
            label_type: Type of label to load (if None, tries both types)
            source: Source of the labels to load (if None, tries both sources)
            
        Returns:
            List of labels or empty list if not found
        """
        image_base = os.path.splitext(image_name)[0]
        
        # Search sources and types based on parameters
        sources = [s.value for s in LabelSource] if source is None else [source.value]
        types = [t.value for t in LabelType] if label_type is None else [label_type.value]
        
        for t in types:
            for s in sources:
                if t == 'bounding_box':
                    label_path = os.path.join(self.dirs['labels']['bounding_boxes'][s], f"{image_base}.txt")
                else:
                    label_path = os.path.join(self.dirs['labels']['object_detection'][s], f"{image_base}.txt")
                
                if os.path.exists(label_path):
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    labels = []
                    for line in lines:
                        values = list(map(float, line.strip().split()))
                        labels.append(tuple(values))
                    
                    return labels
        
        return []
    
    def setup_few_shot_examples(self, classes: List[str]) -> Dict[str, str]:
        """
        Create directories for few-shot examples
        
        Args:
            classes: List of class names
            
        Returns:
            Dictionary mapping class names to their directories
        """
        class_dirs = {}
        for cls in classes:
            cls_dir = os.path.join(self.dirs['few_shot_examples'], cls)
            os.makedirs(cls_dir, exist_ok=True)
            class_dirs[cls] = cls_dir
        
        return class_dirs
    
    def save_few_shot_example(self, class_name: str, image_data: np.ndarray) -> str:
        """
        Save an image as a few-shot example for a class
        
        Args:
            class_name: Name of the class
            image_data: Image data as numpy array
            
        Returns:
            Path to the saved example
        """
        try:
            # Ensure the class directory exists
            cls_dir = os.path.join(self.dirs['few_shot_examples'], class_name)
            os.makedirs(cls_dir, exist_ok=True)
            
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            example_path = os.path.join(cls_dir, f"example_{timestamp}.jpg")
            
            # Save the image
            if not cv2.imwrite(example_path, image_data):
                raise IOError("Failed to write image")
                
            logger.info(f"Saved few-shot example for class '{class_name}': {example_path}")
            return example_path
            
        except Exception as e:
            logger.error(f"Error saving few-shot example: {e}")
            return None
    
    def load_few_shot_examples(self, class_name: str = None) -> Dict[str, List[np.ndarray]]:
        """
        Load few-shot examples for all classes or a specific class
        
        Args:
            class_name: Name of the class to load (if None, loads all classes)
            
        Returns:
            Dictionary mapping class names to lists of example images
        """
        examples = {}
        
        if class_name is not None:
            classes = [class_name]
        else:
            # Get all subdirectories in few_shot_examples
            try:
                classes = [d for d in os.listdir(self.dirs['few_shot_examples']) 
                           if os.path.isdir(os.path.join(self.dirs['few_shot_examples'], d))]
            except FileNotFoundError:
                logger.warning("Few-shot examples directory not found")
                return examples
            except PermissionError:
                logger.error("Permission denied when accessing few-shot examples directory")
                return examples
        
        for cls in classes:
            cls_dir = os.path.join(self.dirs['few_shot_examples'], cls)
            if not os.path.exists(cls_dir):
                logger.warning(f"Class directory does not exist: {cls_dir}")
                continue
                
            examples[cls] = []
            loaded_count = 0
            error_count = 0
            
            for file in os.listdir(cls_dir):
                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(cls_dir, file)
                    try:
                        image = cv2.imread(image_path)
                        if image is not None:
                            examples[cls].append(image)
                            loaded_count += 1
                        else:
                            error_count += 1
                            logger.warning(f"Failed to load image: {image_path}")
                    except Exception as e:
                        error_count += 1
                        logger.error(f"Error loading image {image_path}: {e}")
                        
            logger.info(f"Loaded {loaded_count} examples for class '{cls}' (errors: {error_count})")
                
        return examples
    
    def prepare_yolo_dataset(self, train_ratio: float = 0.8) -> str:
        """
        Prepare YOLO dataset from labeled images
        
        Args:
            train_ratio: Ratio of data to use for training vs validation
            
        Returns:
            Path to the YOLO dataset.yaml file
        """
        try:
            # Clear existing YOLO dataset
            for subdir in ['images/train', 'images/val', 'labels/train', 'labels/val']:
                dir_path = os.path.join(self.dirs['yolo']['main'], subdir)
                if os.path.exists(dir_path):
                    shutil.rmtree(dir_path)
                os.makedirs(dir_path, exist_ok=True)
            
            # Get available classes
            classes = set()
            metadata = self._load_metadata()
            
            # Collect all labeled images
            labeled_images = []
            
            for label_info in metadata['labels'].values():
                if label_info['label_type'] == LabelType.OBJECT_DETECTION.value:
                    # This is a classification + bbox label, suitable for YOLO
                    image_name = label_info['image']
                    label_path = label_info['label_path']
                    
                    # Find the actual image
                    image_path = None
                    for img_type in ['labeled', 'unlabeled']:
                        for root, _, files in os.walk(os.path.join(self.dirs['images'][img_type])):
                            if image_name in files:
                                image_path = os.path.join(root, image_name)
                                break
                        if image_path:
                            break
                    
                    if image_path and os.path.exists(label_path):
                        # Read the label to extract classes
                        try:
                            with open(label_path, 'r') as f:
                                for line in f:
                                    if line.strip():
                                        class_id = int(line.split()[0])
                                        classes.add(class_id)
                            
                            labeled_images.append((image_path, label_path))
                        except Exception as e:
                            logger.warning(f"Error reading label file {label_path}: {e}")
            
            if not labeled_images:
                logger.warning("No labeled images found to prepare YOLO dataset")
                return None
                
            # Split into train/val sets
            random.shuffle(labeled_images)
            split_idx = int(len(labeled_images) * train_ratio)
            train_set = labeled_images[:split_idx]
            val_set = labeled_images[split_idx:]
            
            # Copy images and labels to YOLO dataset
            copy_errors = 0
            
            for dataset, subset in [('train', train_set), ('val', val_set)]:
                for image_path, label_path in subset:
                    try:
                        # Copy image
                        image_name = os.path.basename(image_path)
                        dest_img_path = os.path.join(self.dirs['yolo']['images'][dataset], image_name)
                        shutil.copy2(image_path, dest_img_path)
                        
                        # Copy label
                        label_name = os.path.basename(label_path)
                        dest_label_path = os.path.join(self.dirs['yolo']['labels'][dataset], label_name)
                        shutil.copy2(label_path, dest_label_path)
                    except (OSError, shutil.Error) as e:
                        logger.error(f"Failed to copy files: {e}")
                        copy_errors += 1
            
            if copy_errors:
                logger.warning(f"Encountered {copy_errors} errors while copying dataset files")
                
            # Create dataset.yaml
            yaml_path = os.path.join(self.dirs['yolo']['main'], 'dataset.yaml')
            class_names = ["class_" + str(i) for i in sorted(classes)]
            
            dataset_config = {
                'path': self.dirs['yolo']['main'],
                'train': 'images/train',
                'val': 'images/val',
                'nc': len(classes),
                'names': class_names,
            }
            
            with open(yaml_path, 'w') as f:
                yaml.dump(dataset_config, f, sort_keys=False)
            
            logger.info(f"YOLO dataset prepared: {len(train_set)} training images, {len(val_set)} validation images")
            return yaml_path
            
        except Exception as e:
            logger.error(f"Error preparing YOLO dataset: {e}")
            raise
    
    def get_unprocessed_videos(self) -> List[str]:
        """Get list of unprocessed videos"""
        try:
            video_dir = self.dirs['videos']['unprocessed']
            if not os.path.exists(video_dir):
                logger.warning(f"Unprocessed videos directory not found: {video_dir}")
                return []
                
            videos = [os.path.join(video_dir, f) 
                     for f in os.listdir(video_dir)
                     if os.path.isfile(os.path.join(video_dir, f)) and 
                     f.lower().endswith(('.mp4', '.avi', '.mov'))]
            return videos
            
        except FileNotFoundError:
            logger.warning("Unprocessed videos directory not found")
            return []
        except PermissionError:
            logger.error("Permission denied when accessing unprocessed videos directory")
            return []
    
    def get_processed_videos(self) -> List[str]:
        """Get list of processed videos"""
        try:
            video_dir = self.dirs['videos']['processed']
            if not os.path.exists(video_dir):
                logger.warning(f"Processed videos directory not found: {video_dir}")
                return []
                
            videos = [os.path.join(video_dir, f) 
                     for f in os.listdir(video_dir)
                     if os.path.isfile(os.path.join(video_dir, f)) and 
                     f.lower().endswith(('.mp4', '.avi', '.mov'))]
            return videos
            
        except FileNotFoundError:
            logger.warning("Processed videos directory not found")
            return []
        except PermissionError:
            logger.error("Permission denied when accessing processed videos directory")
            return []
