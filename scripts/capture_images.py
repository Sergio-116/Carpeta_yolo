from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "dataset" / "raw"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Captura imagenes para crear el dataset.")
    parser.add_argument("--class-name", required=True, help="Nombre de la clase, por ejemplo teclado.")
    parser.add_argument("--camera", type=int, default=0, help="Indice de camara. Normalmente 0.")
    parser.add_argument("--width", type=int, default=1280, help="Ancho solicitado a la camara.")
    parser.add_argument("--height", type=int, default=720, help="Alto solicitado a la camara.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    class_dir = OUTPUT_DIR / args.class_name
    class_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara {args.camera}")

    print("Presiona 'c' para capturar, 'q' para salir.")
    count = len(list(class_dir.glob("*.jpg")))

    while True:
        ok, frame = cap.read()
        if not ok:
            print("No se pudo leer frame de la camara.")
            break

        preview = frame.copy()
        cv2.putText(
            preview,
            f"{args.class_name} | capturas: {count} | c=capturar q=salir",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow("Captura dataset YOLO", preview)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("c"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_path = class_dir / f"{args.class_name}_{timestamp}.jpg"
            cv2.imwrite(str(output_path), frame)
            count += 1
            print(f"Guardada: {output_path}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
