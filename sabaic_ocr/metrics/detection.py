from __future__ import annotations

from collections import defaultdict
import torch
from sabaic_ocr.model.box_ops import box_iou_xyxy, xywh_to_xyxy


def targets_to_xyxy(targets):
    if targets.numel()==0: return torch.empty((0,5),dtype=targets.dtype,device=targets.device)
    return torch.cat([xywh_to_xyxy(targets[:,1:5]),targets[:,:1]],dim=1)


def _compute_ap(recall,precision):
    if recall.numel()==0: return 0.0
    mrec=torch.cat([torch.tensor([0.0],device=recall.device),recall,torch.tensor([1.0],device=recall.device)])
    mpre=torch.cat([torch.tensor([1.0],device=precision.device),precision,torch.tensor([0.0],device=precision.device)])
    for i in range(mpre.numel()-2,-1,-1): mpre[i]=torch.maximum(mpre[i],mpre[i+1])
    idx=torch.where(mrec[1:]!=mrec[:-1])[0]
    return float(torch.sum((mrec[idx+1]-mrec[idx])*mpre[idx+1]).item())


def detection_metrics(predictions,ground_truths,num_classes:int,iou_thresholds=tuple(0.50+0.05*i for i in range(10))):
    if len(predictions)!=len(ground_truths): raise ValueError("predictions and ground_truths length mismatch")
    thresholds=list(iou_thresholds); gt_xyxy=[targets_to_xyxy(t.detach().cpu()) for t in ground_truths]; preds=[p.detach().cpu() for p in predictions]
    total_tp=total_fp=total_fn=0; per_class_ap=defaultdict(list)
    for thr_idx,iou_thr in enumerate(thresholds):
        for cls_id in range(num_classes):
            gt_count=0; gt_by_image={}
            for image_id,gt in enumerate(gt_xyxy):
                mask=gt[:,4].long()==cls_id if gt.numel() else torch.zeros((0,),dtype=torch.bool)
                boxes=gt[mask,:4] if gt.numel() else torch.empty((0,4)); gt_by_image[image_id]=boxes; gt_count+=boxes.shape[0]
            if gt_count==0: continue
            pred_rows=[]
            for image_id,pred in enumerate(preds):
                if pred.numel()==0: continue
                mask=pred[:,5].long()==cls_id
                for row in pred[mask]: pred_rows.append((float(row[4]),image_id,row[:4]))
            pred_rows.sort(key=lambda x:x[0],reverse=True)
            matched={i:torch.zeros(len(b),dtype=torch.bool) for i,b in gt_by_image.items()}; tp=torch.zeros(len(pred_rows)); fp=torch.zeros(len(pred_rows))
            for i,(_,image_id,pbox) in enumerate(pred_rows):
                gboxes=gt_by_image[image_id]
                if gboxes.numel()==0: fp[i]=1; continue
                ious=box_iou_xyxy(pbox.unsqueeze(0),gboxes).squeeze(0); best_iou,best_idx=ious.max(dim=0)
                if best_iou>=iou_thr and not matched[image_id][best_idx]: tp[i]=1; matched[image_id][best_idx]=True
                else: fp[i]=1
            if pred_rows:
                tc=tp.cumsum(0); fc=fp.cumsum(0); recall=tc/max(1,gt_count); precision=tc/(tc+fc+1e-9); ap=_compute_ap(recall,precision)
            else: ap=0.0
            per_class_ap[cls_id].append(ap)
            if thr_idx==0:
                cls_tp=int(tp.sum().item()); total_tp+=cls_tp; total_fp+=int(fp.sum().item()); total_fn+=gt_count-cls_tp
    ap50_values=[]; map_values=[]; class_report={}
    for cls_id in range(num_classes):
        aps=per_class_ap.get(cls_id,[])
        if not aps: continue
        ap50=aps[0]; class_map=sum(aps)/len(aps); ap50_values.append(ap50); map_values.append(class_map); class_report[str(cls_id)]={"ap50":ap50,"map50_95":class_map}
    return {"precision_iou50":total_tp/max(1,total_tp+total_fp),"recall_iou50":total_tp/max(1,total_tp+total_fn),"tp":total_tp,"fp":total_fp,"fn":total_fn,"map50":sum(ap50_values)/max(1,len(ap50_values)),"map50_95":sum(map_values)/max(1,len(map_values)),"classes_with_gt":len(map_values),"per_class":class_report}
