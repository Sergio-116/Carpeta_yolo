from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrena YOLO para detectar perifericos.")
    parser.add_argument("--model", default="yolo11n.pt", help="Modelo base: yolo11n.pt, yolo11s.pt, etc.")
    parser.add_argument("--data", default=str(ROOT / "data" / "perifericos.yaml"), help="Archivo data YAML.")
    parser.add_argument("--epochs", type=int, default=50, help="Numero de epocas.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de imagen.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    parser.add_argument("--device", default=None, help="Ej: 0 para GPU, cpu para procesador.")
    parser.add_argument("--name", default="perifericos_yolo", help="Nombre de la corrida.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "name": args.name,
        "project": str(ROOT / "runs" / "detect"),
        "patience": 15,
        "workers": 2,
    }
    if args.device:
        train_kwargs["device"] = args.device
    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
