# Motion-Aware Few-Shot Object Detection System

## Overview
This project implements an integrated motion-aware few-shot object detection system that combines motion detection with few-shot learning for efficient object detection and classification with minimal training data. It's designed to identify moving objects in videos and classify them using few-shot learning techniques, before learning these information with a YOLO architecture, hence empowering users to easily train YOLO object detection models without having to annotate large datasets.

## Key Features
- **Motion Detection**: Identifies moving objects in video streams using background subtraction techniques
- **Few-Shot Learning**: Classifies objects using meta-learning with only a few examples per class
- **Integrated GUI Interface**: Provides tools for video playback, annotation, and model training
- **Annotation Tools**: Allows annotation of classes for bounding boxes generated from motion detection
- **YOLO Integration**: Supports training YOLO-based models for object detection
- **Class Management**: Easy management of object classes across video datasets
- **Batch Processing**: Process multiple videos at once automatically with batch processing
- **Efficient Workflow**: Workflow is highly automated by application such that user only has to interact with easy-to-use UI elements

## Technical Details
- **Meta-Baseline Model**: Uses the [Meta-Baseline](https://github.com/yinboc/few-shot-meta-baseline) architecture for few-shot learning
- **Motion Detection**: Implements MOG2 background subtraction algorithm with configurable parameters
- **Video Buffer System**: Advanced buffer system for smooth video playback
- **PyQt6-Based UI**: Modern user interface built with PyQt6
- **Configuration System**: YAML-based configuration for easy customization

## Installation

### Requirements
- Python 3.8 or higher
- PyQt6
- OpenCV 4.5+
- PyTorch 1.8+
- CUDA-capable GPU recommended for faster processing

### Setup
1. Clone the repository
```bash
git clone https://github.com/yourusername/motion-aware-fewshot-detection.git
cd motion-aware-fewshot-detection
```

2. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies
```bash
pip install -r requirements.txt
```

4. Download pre-trained model weights
```bash
# Create directory for model weights
mkdir -p model_weights

# Download Meta-Baseline weights
# You need to train the Meta-Baseline model using code from:
# https://github.com/yinboc/few-shot-meta-baseline
# and place it in the model_weights directory
```

## Usage

### GUI Mode
Launch the application in GUI mode to access the full interface:

```bash
python main.py --mode gui
```

The GUI provides the following main functionalities:

1. **Video Management**
   - Import individual videos or entire directories of videos
   - Navigate between videos with high precision playback controls such as jumping to a specific frame and skipping intervals of frames

2. **Detection Tab**
   - Configure motion detection sensitivity and frame processing interval
   - Run detection on single videos or batch process multiple videos at once
   - Preview of motion detection is displayed automatically while motion detection is running

3. **Few-Shot Learning Tab**
   - Annotate object classes for bounding boxes from motion detection with simple UI interactions
   - Export support and query images for few-shot learning with a click of a button
   - Train few-shot classifiers with configurable confidence thresholds
   - View classification results from few-shot learning

4. **YOLO Training Tab**
   - Configure YOLO training parameters
   - Prepare training data from annotations generated using motion detection and few-shot learning
   - Train custom YOLO models for specialized detection tasks

### CLI Mode
For headless environments, use the command-line interface:

```bash
python main.py --mode cli
```

### Configuration
The system behavior can be customized through the `config.yaml` file. You can specify:

```bash
python main.py --config custom_config.yaml
```