from __future__ import annotations

from pathlib import Path
import torch
from PIL import Image,ImageDraw
from sabaic_ocr.config import load_classes
from sabaic_ocr.data.dataset import letterbox,pil_to_tensor
from sabaic_ocr.model.decode import postprocess_batch
from sabaic_ocr.model.yolo import SabaicYOLO
from sabaic_ocr.ocr.postprocess import detections_to_text,tensor_detections_to_objects


def load_detector(checkpoint_path,classes_path="config/classes.json",device=None):
    classes_cfg=load_classes(classes_path); dev=torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
    payload=torch.load(checkpoint_path,map_location=dev,weights_only=False); cfg=payload["model_config"]
    model=SabaicYOLO(cfg["num_classes"],cfg.get("width_mult",0.5),cfg.get("depth_mult",0.5)).to(dev); model.load_state_dict(payload["model_state"],strict=True); model.eval()
    if cfg["num_classes"]!=classes_cfg["num_classes"]: raise RuntimeError("Checkpoint classes do not match classes.json")
    return model,cfg,classes_cfg,dev


def infer_pil(image,model,model_cfg,classes_cfg,device,conf_threshold=0.25,iou_threshold=0.45):
    original=image.convert("RGB"); resized,_,meta=letterbox(original,torch.empty((0,5)),int(model_cfg["image_size"])); tensor=pil_to_tensor(resized).unsqueeze(0).to(device)
    with torch.no_grad():
        det=postprocess_batch(model(tensor),model_cfg["anchors"],model_cfg["num_classes"],model_cfg["image_size"],conf_threshold,iou_threshold)[0].cpu()
    id_to_char={int(c["id"]):c["char"] for c in classes_cfg["classes"]}; text=detections_to_text(tensor_detections_to_objects(det),id_to_char,"rtl")
    return {"detections_input_normalized":det,"text":text,"meta":meta}


def restore_box_to_original(box,meta):
    size=float(meta["input_size"]); pad_x,pad_y=meta["pad"]; scale=float(meta["scale"]); old_w,old_h=meta["old_size"]
    x1=(float(box[0])*size-pad_x)/scale; y1=(float(box[1])*size-pad_y)/scale; x2=(float(box[2])*size-pad_x)/scale; y2=(float(box[3])*size-pad_y)/scale
    return max(0,min(old_w,x1)),max(0,min(old_h,y1)),max(0,min(old_w,x2)),max(0,min(old_h,y2))


def annotate_image(image,detections,meta,classes_cfg):
    out=image.convert("RGB").copy(); draw=ImageDraw.Draw(out); names={int(c["id"]):c["name"] for c in classes_cfg["classes"]}
    for row in detections:
        x1,y1,x2,y2=restore_box_to_original(row[:4],meta); score=float(row[4]); cls_id=int(row[5]); draw.rectangle((x1,y1,x2,y2),outline=(255,0,0),width=2); draw.text((x1,max(0,y1-12)),f"{cls_id}:{names.get(cls_id,'?')} {score:.2f}",fill=(255,0,0))
    return out
