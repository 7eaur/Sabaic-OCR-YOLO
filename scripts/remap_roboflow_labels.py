#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path


def parse_roboflow_names(data_yaml: Path) -> list[str]:
    for raw in data_yaml.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("names:"):
            value = line.split(":", 1)[1].strip()
            names = ast.literal_eval(value)
            if not isinstance(names, list) or not all(isinstance(x, str) for x in names):
                raise ValueError("data.yaml names must be a list of strings")
            return names
    raise ValueError(f"Could not find 'names:' in {data_yaml}")


def load_project_ids(classes_json: Path) -> set[int]:
    payload = json.loads(classes_json.read_text(encoding="utf-8"))
    return {int(item["id"]) for item in payload["classes"]}


def build_mapping(exported_names: list[str], valid_project_ids: set[int]) -> dict[int, int]:
    mapping: dict[int, int] = {}
    for export_id, name in enumerate(exported_names):
        try:
            project_id = int(name)
        except ValueError as exc:
            raise ValueError(
                f"Roboflow class name {name!r} is not a numeric project class id. "
                "Use project IDs such as 00, 01, 03, 05, ... as Roboflow class names."
            ) from exc
        if project_id not in valid_project_ids:
            raise ValueError(f"Roboflow class {name!r} maps to unknown project class id {project_id}")
        mapping[export_id] = project_id
    return mapping


def remap_label_file(src: Path, dst: Path, mapping: dict[int, int]) -> int:
    out_lines: list[str] = []
    count = 0
    for lineno, raw in enumerate(src.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            raise ValueError(f"{src}:{lineno}: expected 5 YOLO fields, got {len(parts)}")
        export_id = int(parts[0])
        if export_id not in mapping:
            raise ValueError(f"{src}:{lineno}: unknown exported class id {export_id}")
        coords = [float(x) for x in parts[1:]]
        if not all(0.0 <= x <= 1.0 for x in coords):
            raise ValueError(f"{src}:{lineno}: YOLO coordinates must be normalized to [0,1]")
        out_lines.append(" ".join([str(mapping[export_id]), *parts[1:]]))
        count += 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")
    return count


def write_project_yaml(out_root: Path, num_classes: int) -> None:
    names = [f"{i:02d}" for i in range(num_classes)]
    text = (
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {num_classes}\n"
        f"names: {names!r}\n"
    )
    (out_root / "data.yaml").write_text(text, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Remap Roboflow-exported YOLO class indices back to the project's fixed class IDs."
    )
    p.add_argument("--dataset", required=True, help="Roboflow dataset root containing data.yaml")
    p.add_argument("--output", required=True, help="Output directory; original export is never modified")
    p.add_argument("--classes", default="config/classes.json")
    args = p.parse_args()

    src_root = Path(args.dataset).resolve()
    out_root = Path(args.output).resolve()
    data_yaml = src_root / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Missing {data_yaml}")
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    exported_names = parse_roboflow_names(data_yaml)
    valid_project_ids = load_project_ids(Path(args.classes))
    mapping = build_mapping(exported_names, valid_project_ids)

    print("Roboflow export -> project class mapping")
    for export_id, project_id in mapping.items():
        print(f"  {export_id:2d} -> {project_id:02d}")

    image_count = 0
    label_count = 0
    box_count = 0
    for split in ("train", "valid", "test"):
        src_images = src_root / split / "images"
        src_labels = src_root / split / "labels"
        if src_images.exists():
            dst_images = out_root / split / "images"
            dst_images.mkdir(parents=True, exist_ok=True)
            for image in src_images.iterdir():
                if image.is_file():
                    shutil.copy2(image, dst_images / image.name)
                    image_count += 1
        if src_labels.exists():
            for label in src_labels.glob("*.txt"):
                box_count += remap_label_file(
                    label, out_root / split / "labels" / label.name, mapping
                )
                label_count += 1

    # Keep provenance/readme files when present, but replace data.yaml with project-wide 30-class definition.
    for name in ("README.roboflow.txt", "README.dataset.txt"):
        pth = src_root / name
        if pth.exists():
            shutil.copy2(pth, out_root / name)
    write_project_yaml(out_root, len(valid_project_ids))

    print(f"images: {image_count}")
    print(f"label_files: {label_count}")
    print(f"boxes: {box_count}")
    print(f"output: {out_root}")
    print("status: OK - labels now use fixed project class IDs 0..29")


if __name__ == "__main__":
    main()
