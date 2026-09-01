#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from sabaic_ocr.config import load_classes


def main() -> None:
    p = argparse.ArgumentParser(description="Validate that the supplied TTF renders every configured class.")
    p.add_argument("--font", default="assets/fonts/NotoSansOldSouthArabian-Regular.ttf")
    p.add_argument("--classes", default="config/classes.json")
    p.add_argument("--size", type=int, default=96)
    args = p.parse_args()

    font_path = Path(args.font)
    if not font_path.exists():
        raise SystemExit(f"Font not found: {font_path}")

    cfg = load_classes(args.classes)
    font = ImageFont.truetype(str(font_path), args.size)
    report = {
        "font": str(font_path),
        "sha256": hashlib.sha256(font_path.read_bytes()).hexdigest(),
        "classes": cfg["num_classes"],
        "empty_glyphs": [],
        "duplicate_rasters": [],
        "glyphs": [],
    }

    raster_hashes: dict[str, list[int]] = {}
    for item in cfg["classes"]:
        ch = item["char"]
        canvas = Image.new("L", (args.size * 2, args.size * 2), 0)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), ch, font=font)
        w = max(0, bbox[2] - bbox[0])
        h = max(0, bbox[3] - bbox[1])
        draw.text((10 - bbox[0], 10 - bbox[1]), ch, font=font, fill=255)
        visible = canvas.getbbox()
        if visible is None or w == 0 or h == 0:
            report["empty_glyphs"].append(item["id"])
            digest = ""
        else:
            crop = canvas.crop(visible)
            digest = hashlib.sha256(crop.tobytes()).hexdigest()
            raster_hashes.setdefault(digest, []).append(int(item["id"]))
        report["glyphs"].append({
            "id": item["id"],
            "unicode": item["unicode"],
            "bbox": [int(v) for v in bbox],
            "width": w,
            "height": h,
        })

    report["duplicate_rasters"] = [ids for ids in raster_hashes.values() if len(ids) > 1]
    report["valid"] = not report["empty_glyphs"] and not report["duplicate_rasters"]
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
