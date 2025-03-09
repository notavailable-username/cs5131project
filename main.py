import argparse
import yaml
from cli import run_cli
from gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
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

def main():
    parser = argparse.ArgumentParser(description="Integrated Motion-Aware Few-Shot Object Detection System")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui", help="Launch mode: gui or cli")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    print(f"Starting application in {args.mode.upper()} mode")
    config = load_config(args.config)

    if args.mode == "gui":
        print("Initializing GUI application...")
        app = QApplication(sys.argv)
        window = MainWindow(config)
        window.show()
        print("GUI initialized and displayed. Running application event loop.")
        sys.exit(app.exec())
    else:
        print("Running command-line interface...")
        run_cli(config)

if __name__ == "__main__":
    print("Motion-Aware Few-Shot Object Detection System")
    print("=" * 50)
    main()
