from __future__ import annotations

import argparse
import shutil
import unicodedata
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "dataset" / "Pendientes por etiqueta"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_MAP = {
    "teclado": 0,
    "mouse": 1,
    "monitor": 2,
    "impresora": 3,
    "parlantes": 4,
    "webcam": 5,
    "microfono": 6,
    "audifonos": 7,
    "memoria_usb": 8,
    "memoria usb": 8,
    "router": 9,
    "ruoter": 9,
}


def normalize_text(value: str) -> str:
    ascii_text = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return " ".join(ascii_text.lower().replace("_", " ").split())


def class_id_from_folder(folder_name: str) -> int:
    normalized = normalize_text(folder_name)
    for key, class_id in CLASS_MAP.items():
        if key in normalized:
            return class_id
    raise ValueError(f"No se pudo mapear la carpeta a una clase YOLO: {folder_name}")


def split_from_folder(folder_name: str) -> str:
    normalized = normalize_text(folder_name)
    if normalized == "train":
        return "train"
    if normalized == "val":
        return "val"
    raise ValueError(f"El split debe llamarse Train o Val, pero se encontro: {folder_name}")


def safe_stem(value: str) -> str:
    normalized = normalize_text(value)
    return normalized.replace(" ", "_")


def find_object_bbox(image_path: Path) -> tuple[float, float, float, float]:
    image = cv2.imdecode(np.fromfile(str(image_path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"No se pudo leer la imagen: {image_path}")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha_mask = None
    elif image.shape[2] == 4:
        bgr = image[:, :, :3]
        alpha_mask = image[:, :, 3] > 10
    else:
        bgr = image[:, :, :3]
        alpha_mask = None

    height, width = bgr.shape[:2]
    if width == 0 or height == 0:
        return 0.5, 0.5, 1.0, 1.0

    if alpha_mask is not None and alpha_mask.any():
        mask = alpha_mask.astype(np.uint8) * 255
    else:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.int16)
        corner_size = max(5, min(width, height) // 20)
        corners = np.concatenate(
            [
                rgb[:corner_size, :corner_size].reshape(-1, 3),
                rgb[:corner_size, -corner_size:].reshape(-1, 3),
                rgb[-corner_size:, :corner_size].reshape(-1, 3),
                rgb[-corner_size:, -corner_size:].reshape(-1, 3),
            ],
            axis=0,
        )
        background = np.median(corners, axis=0)
        color_distance = np.linalg.norm(rgb - background, axis=2)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 140)
        mask = ((color_distance > 28) | (edges > 0)).astype(np.uint8) * 255

    kernel_size = max(3, min(width, height) // 80)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)

    points = cv2.findNonZero(mask)
    if points is None:
        return 0.5, 0.5, 0.92, 0.92

    x, y, box_w, box_h = cv2.boundingRect(points)
    area_ratio = (box_w * box_h) / float(width * height)
    if area_ratio < 0.03:
        return 0.5, 0.5, 0.92, 0.92

    padding_x = int(width * 0.025)
    padding_y = int(height * 0.025)
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(width, x + box_w + padding_x)
    y2 = min(height, y + box_h + padding_y)

    x_center = ((x1 + x2) / 2) / width
    y_center = ((y1 + y2) / 2) / height
    norm_w = (x2 - x1) / width
    norm_h = (y2 - y1) / height
    return x_center, y_center, norm_w, norm_h


def iter_images(path: Path) -> list[Path]:
    return sorted(
        file
        for file in path.rglob("*")
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def clean_output_dirs() -> None:
    for folder in (
        ROOT / "dataset" / "images" / "train",
        ROOT / "dataset" / "images" / "val",
        ROOT / "dataset" / "labels" / "train",
        ROOT / "dataset" / "labels" / "val",
    ):
        folder.mkdir(parents=True, exist_ok=True)
        for file in folder.iterdir():
            if file.is_file() and file.name != ".gitkeep":
                file.unlink()


def prepare_dataset(source: Path, limit_train: int | None, limit_val: int | None) -> dict[str, int]:
    output_images = ROOT / "dataset" / "images"
    output_labels = ROOT / "dataset" / "labels"
    stats: dict[str, int] = {}

    for class_dir in sorted(path for path in source.iterdir() if path.is_dir()):
        class_id = class_id_from_folder(class_dir.name)
        class_slug = safe_stem(class_dir.name)

        for split_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            split = split_from_folder(split_dir.name)
            limit = limit_train if split == "train" else limit_val
            images = iter_images(split_dir)
            if limit is not None:
                images = images[:limit]

            (output_images / split).mkdir(parents=True, exist_ok=True)
            (output_labels / split).mkdir(parents=True, exist_ok=True)

            written = 0
            skipped = 0
            for index, image_path in enumerate(images, start=1):
                extension = image_path.suffix.lower()
                target_name = f"{class_slug}_{split}_{index:04d}{extension}"
                target_image = output_images / split / target_name
                target_label = output_labels / split / f"{Path(target_name).stem}.txt"

                try:
                    x_center, y_center, width, height = find_object_bbox(image_path)
                except ValueError as error:
                    skipped += 1
                    print(f"Saltada: {error}")
                    continue

                shutil.copy2(image_path, target_image)
                target_label.write_text(
                    f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n",
                    encoding="utf-8",
                )
                written += 1

            stats[f"{class_dir.name}/{split}"] = written
            if skipped:
                stats[f"{class_dir.name}/{split} saltadas"] = skipped

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepara imagenes y etiquetas YOLO automaticamente.")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE), help="Carpeta con clases y subcarpetas Train/Val.")
    parser.add_argument("--limit-train", type=int, default=None, help="Maximo de imagenes train por clase.")
    parser.add_argument("--limit-val", type=int, default=None, help="Maximo de imagenes val por clase.")
    parser.add_argument("--clean-output", action="store_true", help="Limpia images/labels antes de generar.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    if not source.exists():
        raise FileNotFoundError(f"No existe la carpeta: {source}")

    if args.clean_output:
        clean_output_dirs()

    stats = prepare_dataset(source, args.limit_train, args.limit_val)
    print("Dataset YOLO preparado:")
    for key, count in stats.items():
        print(f"  {key}: {count} imagenes etiquetadas")


if __name__ == "__main__":
    main()
