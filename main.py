import argparse
import yaml
from cli import run_cli
from gui.main_window import MainWindow
from PyQt6.QtWidgets import QApplication
import sys

def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Integrated Motion-Aware Few-Shot Object Detection System")
    parser.add_argument("--mode", choices=["gui", "cli"], default="gui", help="Launch mode: gui or cli")
    args = parser.parse_args()

    config = load_config()

    if args.mode == "gui":
        app = QApplication(sys.argv)
        window = MainWindow(config)
        window.show()
        sys.exit(app.exec())
    else:
        run_cli(config)

if __name__ == "__main__":
    main()
