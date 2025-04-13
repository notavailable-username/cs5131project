import argparse
import yaml
from cli import InteractiveCLI
from gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QIcon
import sys
import os

def load_config(config_path="config.yaml"):
    print(f"Loading configuration from {os.path.abspath(config_path)}")
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
            print(f"Configuration loaded successfully. Found {len(config)} main sections.")
            return config
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found: {config_path}")
        print("Using default configuration.")
        return {}
    except yaml.YAMLError as e:
        print(f"ERROR: Failed to parse configuration file: {e}")
        print("Using default configuration.")
        return {}

def ensure_directory_structure():
    """Ensure the required directory structure exists, creating directories if needed."""
    base_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "datasets")
    
    # Define the required directory structure
    required_dirs = [
        base_dir,
        os.path.join(base_dir, "annotations"),
        os.path.join(base_dir, "annotations", "videos"),
        os.path.join(base_dir, "annotations", "images"),
        os.path.join(base_dir, "video_configs"),
        os.path.join(base_dir, "videos"),
        os.path.join(base_dir, "images"),
        os.path.join(base_dir, "supports"),
        os.path.join(base_dir, "queries"),
        os.path.join(base_dir, "uncertain"),
        # YOLO directory structure - updated to correct format
        os.path.join(base_dir, "yolo"),
        os.path.join(base_dir, "yolo", "images"),
        os.path.join(base_dir, "yolo", "images", "train"),
        os.path.join(base_dir, "yolo", "images", "val"),
        os.path.join(base_dir, "yolo", "images", "test"),
        os.path.join(base_dir, "yolo", "labels"),
        os.path.join(base_dir, "yolo", "labels", "train"),
        os.path.join(base_dir, "yolo", "labels", "val"),
        os.path.join(base_dir, "yolo", "labels", "test"),
        os.path.join(base_dir, "yolo", "config")
    ]
    
    # Check and create directories if needed
    created_dirs = []
    for directory in required_dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            created_dirs.append(directory)
    
    # Print information about created directories
    if created_dirs:
        print("Created the following directories:")
        for directory in created_dirs:
            print(f"  - {os.path.relpath(directory, os.path.dirname(os.path.abspath(__file__)))}")
    else:
        print("All required directories already exist.")

def main():
    parser = argparse.ArgumentParser(description="Integrated Motion-Aware Few-Shot Object Detection System")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui", help="Launch mode: gui or cli")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    print(f"Starting application in {args.mode.upper()} mode")
    config = load_config(args.config)
    
    # Ensure the directory structure exists
    print("Checking required directory structure...")
    ensure_directory_structure()

    if args.mode == "gui":
        print("Initializing GUI application...")
        app = QApplication(sys.argv)
        app.setWindowIcon(QIcon('/gui/app_icon.png'))

        window = MainWindow(config)
        window.show()
        print("GUI initialized and displayed. Running application event loop.")
        sys.exit(app.exec())
    else:
        print("Running command-line interface...")
        cli_app = InteractiveCLI(config)
        cli_app.run()

if __name__ == "__main__":
    print("Motion-Aware Few-Shot Object Detection System")
    print("=" * 50)
    main()
