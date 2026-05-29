from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]


def parse_source(value: str) -> int | str:
    if value.isdigit():
        return int(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ejecuta deteccion YOLO con imagen, video o camara.")
    parser.add_argument("--weights", required=True, help="Ruta a best.pt, modelo ONNX o carpeta NCNN.")
    parser.add_argument("--source", default="0", help="0 para webcam, ruta de imagen/video o stream.")
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de inferencia.")
    parser.add_argument("--conf", type=float, default=0.35, help="Confianza minima.")
    parser.add_argument("--show", action="store_true", help="Muestra ventana con resultados.")
    parser.add_argument("--save", action="store_true", help="Guarda resultados en runs/detect.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    model.predict(
        source=parse_source(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        show=args.show,
        save=args.save,
        project=str(ROOT / "runs" / "detect"),
        name="predict",
    )


if __name__ == "__main__":
    main()
