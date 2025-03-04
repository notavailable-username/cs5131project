import argparse
from models.yolo_trainer import YOLOTrainer

def run_cli(config):
    parser = argparse.ArgumentParser(description="Command Line Interface for Model Training")
    subparsers = parser.add_subparsers(dest="command", help="Sub-command help")

    # YOLO training sub-command
    yolo_parser = subparsers.add_parser("train_yolo", help="Train YOLOv11")
    yolo_parser.add_argument("--pretrained", action="store_true", help="Use pretrained weights")

    args = parser.parse_args()

    if args.command == "train_yolo":
        trainer = YOLOTrainer(config["yolo"])
        trainer.train(use_pretrained=args.pretrained)
    else:
        parser.print_help()

if __name__ == "__main__":
    run_cli({})
