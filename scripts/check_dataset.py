from __future__ import annotations

from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "data" / "perifericos.yaml"
DATASET = ROOT / "dataset"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_classes() -> dict[int, str]:
    with DATA_YAML.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    return {int(k): v for k, v in data["names"].items()}


def image_files(split: str) -> list[Path]:
    image_dir = DATASET / "images" / split
    return sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def label_path_for(image_path: Path, split: str) -> Path:
    return DATASET / "labels" / split / f"{image_path.stem}.txt"


def validate_label(label_path: Path, class_ids: set[int]) -> list[str]:
    errors: list[str] = []
    if not label_path.exists():
        return [f"Falta etiqueta: {label_path}"]

    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"{label_path}:{line_number} debe tener 5 valores")
            continue
        try:
            class_id = int(parts[0])
            values = [float(value) for value in parts[1:]]
        except ValueError:
            errors.append(f"{label_path}:{line_number} contiene valores invalidos")
            continue
        if class_id not in class_ids:
            errors.append(f"{label_path}:{line_number} clase {class_id} no existe")
        if any(value < 0 or value > 1 for value in values):
            errors.append(f"{label_path}:{line_number} coordenadas fuera de 0..1")
    return errors


def main() -> None:
    classes = load_classes()
    class_ids = set(classes.keys())
    total_errors: list[str] = []

    print("Clases configuradas:")
    for class_id, name in classes.items():
        print(f"  {class_id}: {name}")

    for split in ("train", "val"):
        images = image_files(split)
        print(f"\n{split}: {len(images)} imagenes")
        for image_path in images:
            total_errors.extend(validate_label(label_path_for(image_path, split), class_ids))

    if total_errors:
        print("\nErrores encontrados:")
        for error in total_errors:
            print(f"- {error}")
        raise SystemExit(1)

    print("\nDataset listo para entrenar.")


if __name__ == "__main__":
    main()
