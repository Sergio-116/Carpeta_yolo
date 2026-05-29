from __future__ import annotations

import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exporta un modelo YOLO para Raspberry Pi.")
    parser.add_argument("--weights", required=True, help="Ruta al archivo best.pt.")
    parser.add_argument(
        "--format",
        default="ncnn",
        choices=("ncnn", "onnx", "openvino"),
        help="Formato de exportacion. NCNN suele ser buena opcion en Raspberry Pi.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Tamano de imagen para exportar.")
    parser.add_argument("--half", action="store_true", help="Exporta FP16 si el formato lo soporta.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.weights)
    output = model.export(format=args.format, imgsz=args.imgsz, half=args.half)
    print(f"Modelo exportado: {output}")


if __name__ == "__main__":
    main()
