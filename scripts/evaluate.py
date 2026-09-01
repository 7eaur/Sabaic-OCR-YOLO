#!/usr/bin/env python3
from __future__ import annotations

import argparse,json
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from sabaic_ocr.config import load_classes
from sabaic_ocr.data.dataset import YoloCharacterDataset,yolo_collate
from sabaic_ocr.metrics.detection import detection_metrics
from sabaic_ocr.metrics.ocr import evaluate_corpus
from sabaic_ocr.model.decode import postprocess_batch
from sabaic_ocr.ocr.postprocess import detections_to_text,tensor_detections_to_objects
from sabaic_ocr.training.engine import build_model


def main():
    p=argparse.ArgumentParser(description="Evaluate detection + OCR on untouched real test split."); p.add_argument("--checkpoint",required=True); p.add_argument("--images",default="data/real/images/test"); p.add_argument("--labels",default="data/real/labels/test"); p.add_argument("--transcripts",default="data/real/transcripts/test"); p.add_argument("--classes",default="config/classes.json"); p.add_argument("--batch-size",type=int,default=8); p.add_argument("--conf",type=float,default=0.25); p.add_argument("--iou",type=float,default=0.45); p.add_argument("--output",default="outputs/evaluation/test_metrics.json"); args=p.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); payload=torch.load(args.checkpoint,map_location=device,weights_only=False); cfg=payload["model_config"]; classes=load_classes(args.classes); model=build_model(cfg).to(device); model.load_state_dict(payload["model_state"],strict=True); model.eval()
    ds=YoloCharacterDataset(args.images,args.labels,cfg["num_classes"],cfg["image_size"],None,True); loader=DataLoader(ds,batch_size=args.batch_size,shuffle=False,num_workers=2,collate_fn=yolo_collate); preds=[]; gts=[]; pairs=[]; id_to_char={int(c["id"]):c["char"] for c in classes["classes"]}
    with torch.no_grad():
        for images,targets,metas in loader:
            detections=postprocess_batch(model(images.to(device)),cfg["anchors"],cfg["num_classes"],cfg["image_size"],args.conf,args.iou)
            for det,target,meta in zip(detections,targets,metas):
                det=det.cpu(); preds.append(det); gts.append(target.cpu()); pred_text=detections_to_text(tensor_detections_to_objects(det),id_to_char,"rtl"); gt_path=Path(args.transcripts)/f"{Path(meta['image_path']).stem}.txt"
                if not gt_path.exists(): raise FileNotFoundError(f"Missing test transcript: {gt_path}")
                pairs.append((gt_path.read_text(encoding="utf-8").strip(),pred_text))
    report={"checkpoint":args.checkpoint,"test_images":len(ds),"confidence_threshold":args.conf,"nms_iou_threshold":args.iou,"detection":detection_metrics(preds,gts,cfg["num_classes"]),"ocr":evaluate_corpus(pairs)}; out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
