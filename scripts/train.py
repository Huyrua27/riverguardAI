"""Train the detection model (thin wrapper around Ultralytics).

Usage:
    python scripts/train.py --data configs/data.yaml --epochs 100
"""
from __future__ import annotations

import argparse


def main() -> None:
    from riverguard.utils.console import utf8_console

    utf8_console()
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=1280)  # small/far waste → high-res
    args = parser.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.model)
    model.train(data=args.data, epochs=args.epochs, imgsz=args.imgsz)


if __name__ == "__main__":
    main()
