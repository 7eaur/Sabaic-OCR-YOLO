#!/usr/bin/env python3
from __future__ import annotations

import argparse,json,random
from pathlib import Path
from sabaic_ocr.config import load_classes
from sabaic_ocr.data.dataset import list_images
from sabaic_ocr.data.labels import draw_label_preview,validate_dataset


def main():
    p=argparse.ArgumentParser(); p.add_argument("--images",required=True); p.add_argument("--labels",required=True); p.add_argument("--classes",default="config/classes.json"); p.add_argument("--preview-dir",default=""); p.add_argument("--preview-count",type=int,default=30); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
    cfg=load_classes(args.classes); report=validate_dataset(args.images,args.labels,cfg["num_classes"]); print(json.dumps(report,indent=2))
    if args.preview_dir:
        names={c["id"]:c["name"] for c in cfg["classes"]}; images=list_images(args.images); random.Random(args.seed).shuffle(images)
        for image_path in images[:args.preview_count]:
            label_path=Path(args.labels)/f"{image_path.stem}.txt"
            if label_path.exists(): draw_label_preview(image_path,label_path,Path(args.preview_dir)/f"{image_path.stem}.jpg",names)
        print(f"Previews saved to: {args.preview_dir}")
    if not report["valid"]: raise SystemExit(2)

if __name__=="__main__": main()
