#!/usr/bin/env python3
from __future__ import annotations

import argparse,json
from pathlib import Path
import numpy as np
from sabaic_ocr.data.dataset import read_yolo_labels,list_images


def wh_iou_np(wh,centers):
    wh=wh[:,None,:]; centers=centers[None,:,:]; inter=np.minimum(wh,centers).prod(axis=2); union=wh.prod(axis=2)+centers.prod(axis=2)-inter; return inter/np.maximum(union,1e-9)


def kmeans_iou(box_wh,k,seed,iterations=100):
    rng=np.random.default_rng(seed)
    if len(box_wh)<k: raise ValueError(f"Need at least {k} boxes; found {len(box_wh)}")
    centers=box_wh[rng.choice(len(box_wh),size=k,replace=False)].copy()
    for _ in range(iterations):
        assignment=np.argmax(wh_iou_np(box_wh,centers),axis=1); new=centers.copy()
        for j in range(k):
            pts=box_wh[assignment==j]
            if len(pts): new[j]=np.median(pts,axis=0)
        if np.allclose(new,centers,atol=0.1): break
        centers=new
    return centers


def main():
    p=argparse.ArgumentParser(); p.add_argument("--images",required=True); p.add_argument("--labels",required=True); p.add_argument("--image-size",type=int,default=640); p.add_argument("--num-classes",type=int,default=30); p.add_argument("--k",type=int,default=9); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
    wh=[]
    for img in list_images(args.images):
        labels=read_yolo_labels(Path(args.labels)/f"{img.stem}.txt",args.num_classes)
        if labels.numel(): wh.extend((labels[:,3:5].numpy()*args.image_size).tolist())
    centers=kmeans_iou(np.asarray(wh,dtype=np.float32),args.k,args.seed); centers=centers[np.argsort(centers[:,0]*centers[:,1])]; result=centers.round(2).tolist(); print(json.dumps({"anchors":[result[:3],result[3:6],result[6:9]]},indent=2))

if __name__=="__main__": main()
