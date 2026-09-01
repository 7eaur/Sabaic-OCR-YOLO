#!/usr/bin/env python3
from __future__ import annotations

import argparse,json
from pathlib import Path
from PIL import Image
from sabaic_ocr.inference import annotate_image,infer_pil,load_detector,restore_box_to_original


def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",required=True); p.add_argument("--image",required=True); p.add_argument("--classes",default="config/classes.json"); p.add_argument("--output-dir",default="outputs/inference"); p.add_argument("--conf",type=float,default=0.25); p.add_argument("--iou",type=float,default=0.45); args=p.parse_args()
    model,cfg,classes,device=load_detector(args.checkpoint,args.classes); image=Image.open(args.image).convert("RGB"); result=infer_pil(image,model,cfg,classes,device,args.conf,args.iou); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); stem=Path(args.image).stem
    annotate_image(image,result["detections_input_normalized"],result["meta"],classes).save(out/f"{stem}_boxes.jpg",quality=92)
    rows=[]
    for row in result["detections_input_normalized"]:
        rows.append({"class_id":int(row[5]),"score":float(row[4]),"box_xyxy_original":list(restore_box_to_original(row[:4],result["meta"]))})
    payload={"image":str(args.image),"checkpoint":str(args.checkpoint),"device":str(device),"text":result["text"],"detections":rows}; (out/f"{stem}.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding="utf-8"); (out/f"{stem}.txt").write_text(result["text"],encoding="utf-8"); print(json.dumps(payload,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
