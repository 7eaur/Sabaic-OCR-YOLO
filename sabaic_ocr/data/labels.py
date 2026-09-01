from __future__ import annotations

from collections import Counter
from pathlib import Path
from PIL import Image, ImageDraw
from .dataset import list_images, read_yolo_labels


def validate_dataset(images_dir, labels_dir, num_classes: int) -> dict:
    images_dir, labels_dir = Path(images_dir), Path(labels_dir)
    images = list_images(images_dir)
    errors, class_counts = [], Counter()
    labeled_images = box_count = 0
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            errors.append(f"missing label: {label_path}")
            continue
        try:
            targets = read_yolo_labels(label_path, num_classes)
        except Exception as exc:
            errors.append(str(exc))
            continue
        labeled_images += 1
        box_count += int(targets.shape[0])
        for cls in targets[:,0].tolist():
            class_counts[int(cls)] += 1
    image_stems = {p.stem for p in images}
    orphan_labels = [str(p) for p in labels_dir.glob("*.txt") if p.stem not in image_stems] if labels_dir.exists() else []
    return {"images":len(images),"labeled_images":labeled_images,"boxes":box_count,"class_counts":dict(sorted(class_counts.items())),"errors":errors,"orphan_labels":orphan_labels,"valid":not errors and not orphan_labels and len(images)==labeled_images}


def draw_label_preview(image_path, label_path, output_path, class_names):
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    w,h = image.size
    targets = read_yolo_labels(label_path, len(class_names))
    for row in targets.tolist():
        cls,cx,cy,bw,bh = row
        cls = int(cls)
        x1,y1,x2,y2 = (cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h
        draw.rectangle((x1,y1,x2,y2), outline=(255,0,0), width=2)
        draw.text((x1,max(0,y1-12)), class_names.get(cls,str(cls)), fill=(255,0,0))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
