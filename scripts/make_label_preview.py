from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "dataset" / "images" / "train"
LABELS = ROOT / "dataset" / "labels" / "train"
OUTPUT = ROOT / "dataset" / "preview_auto_labels.jpg"
CLASS_NAMES = {
    0: "teclado",
    1: "mouse",
    2: "monitor",
    3: "impresora",
    4: "parlantes",
    5: "webcam",
    6: "microfono",
    7: "audifonos",
    8: "memoria_usb",
    9: "router",
}


def read_image(path: Path) -> np.ndarray | None:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def draw_label(image: np.ndarray, label_path: Path) -> np.ndarray:
    height, width = image.shape[:2]
    if not label_path.exists():
        return image

    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x_center, y_center, box_w, box_h = [float(value) for value in parts[1:]]
        x1 = int((x_center - box_w / 2) * width)
        y1 = int((y_center - box_h / 2) * height)
        x2 = int((x_center + box_w / 2) * width)
        y2 = int((y_center + box_h / 2) * height)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 0), 3)
        cv2.putText(
            image,
            CLASS_NAMES.get(class_id, str(class_id)),
            (max(0, x1), max(25, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 210, 0),
            2,
            cv2.LINE_AA,
        )
    return image


def main() -> None:
    samples: list[np.ndarray] = []
    seen_prefixes: set[str] = set()

    for image_path in sorted(IMAGES.iterdir()):
        if not image_path.is_file() or image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
            continue
        prefix = image_path.stem.split("_train_")[0]
        if prefix in seen_prefixes:
            continue
        image = read_image(image_path)
        if image is None:
            continue
        image = draw_label(image, LABELS / f"{image_path.stem}.txt")
        image = cv2.resize(image, (260, 260), interpolation=cv2.INTER_AREA)
        samples.append(image)
        seen_prefixes.add(prefix)
        if len(samples) == 10:
            break

    if not samples:
        raise SystemExit("No se encontraron muestras para la vista previa.")

    while len(samples) < 10:
        samples.append(np.full((260, 260, 3), 255, dtype=np.uint8))

    rows = [np.hstack(samples[0:5]), np.hstack(samples[5:10])]
    canvas = np.vstack(rows)
    cv2.imencode(".jpg", canvas)[1].tofile(str(OUTPUT))
    print(f"Vista previa creada: {OUTPUT}")


if __name__ == "__main__":
    main()
