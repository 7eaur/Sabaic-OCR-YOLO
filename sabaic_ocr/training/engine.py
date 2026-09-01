from __future__ import annotations

import math,random,time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader
from sabaic_ocr.data.augment import PhotometricAugment
from sabaic_ocr.data.dataset import YoloCharacterDataset,yolo_collate,list_images
from sabaic_ocr.model.loss import YoloLoss
from sabaic_ocr.model.yolo import SabaicYOLO
from sabaic_ocr.training.checkpoint import load_checkpoint,save_checkpoint


def set_seed(seed:int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def build_model(model_cfg):
    return SabaicYOLO(model_cfg["num_classes"],model_cfg.get("width_mult",0.5),model_cfg.get("depth_mult",0.5))


def build_loader(images_dir,labels_dir,model_cfg,batch_size,num_workers,augment,shuffle):
    ds=YoloCharacterDataset(images_dir,labels_dir,model_cfg["num_classes"],model_cfg["image_size"],PhotometricAugment() if augment else None)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,pin_memory=torch.cuda.is_available(),collate_fn=yolo_collate,drop_last=False)


def cosine_lambda(epoch,total_epochs,warmup_epochs):
    if epoch<warmup_epochs: return max(0.05,(epoch+1)/max(1,warmup_epochs))
    p=(epoch-warmup_epochs)/max(1,total_epochs-warmup_epochs)
    return 0.05+0.95*0.5*(1.0+math.cos(math.pi*p))


def _run_epoch(model,loader,criterion,device,optimizer=None,scaler=None,amp=True):
    train=optimizer is not None; model.train(train); total=box=obj=cls=0.0; positives=batches=0
    for images,targets in loader:
        images=images.to(device,non_blocking=True); targets=[t.to(device,non_blocking=True) for t in targets]
        if train: optimizer.zero_grad(set_to_none=True)
        use_amp=bool(amp and device.type=="cuda")
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type,enabled=use_amp):
                loss=criterion(model(images),targets)
            if train:
                if scaler is not None and use_amp:
                    scaler.scale(loss.total).backward(); scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); scaler.step(optimizer); scaler.update()
                else:
                    loss.total.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),10.0); optimizer.step()
        total+=float(loss.total.detach().item()); box+=float(loss.box.item()); obj+=float(loss.obj.item()); cls+=float(loss.cls.item()); positives+=loss.positives; batches+=1
    d=max(1,batches)
    return {"loss":total/d,"box_loss":box/d,"obj_loss":obj/d,"cls_loss":cls/d,"positives":positives}


def train_detector(train_cfg,model_cfg,init_checkpoint=None,require_min_train_images=None):
    set_seed(int(train_cfg.get("seed",42)))
    if require_min_train_images is not None:
        count=len(list_images(train_cfg["images_dir"]))
        if count<require_min_train_images: raise RuntimeError(f"Fine-tuning requires at least {require_min_train_images} real train images; found {count}.")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=build_model(model_cfg).to(device)
    criterion=YoloLoss(model_cfg["anchors"],model_cfg["num_classes"],model_cfg["image_size"]).to(device)
    optimizer=torch.optim.SGD(model.parameters(),lr=float(train_cfg["learning_rate"]),momentum=float(train_cfg.get("momentum",0.937)),weight_decay=float(train_cfg.get("weight_decay",5e-4)),nesterov=True)
    epochs=int(train_cfg["epochs"]); warmup=int(train_cfg.get("warmup_epochs",3)); scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda e:cosine_lambda(e,epochs,warmup))
    train_loader=build_loader(train_cfg["images_dir"],train_cfg["labels_dir"],model_cfg,int(train_cfg["batch_size"]),int(train_cfg.get("num_workers",2)),bool(train_cfg.get("augmentation",True)),True)
    val_loader=build_loader(train_cfg["val_images_dir"],train_cfg["val_labels_dir"],model_cfg,int(train_cfg["batch_size"]),int(train_cfg.get("num_workers",2)),False,False)
    start_epoch=0; best_val=float("inf")
    resume=train_cfg.get("resume") or ""
    if resume:
        payload=load_checkpoint(resume,model,optimizer,scheduler,device); start_epoch=int(payload.get("epoch",-1))+1; best_val=float(payload.get("best_metric",best_val))
    elif init_checkpoint:
        payload=load_checkpoint(init_checkpoint,model,device=device); ckpt_classes=payload.get("model_config",{}).get("num_classes")
        if ckpt_classes is not None and int(ckpt_classes)!=int(model_cfg["num_classes"]): raise RuntimeError("Checkpoint num_classes does not match current model config.")
    amp_enabled=bool(train_cfg.get("amp",True) and device.type=="cuda"); scaler=torch.amp.GradScaler("cuda",enabled=amp_enabled) if device.type=="cuda" else None
    checkpoint_dir=Path(train_cfg["checkpoint_dir"]); checkpoint_dir.mkdir(parents=True,exist_ok=True); history=[]; save_every=int(train_cfg.get("save_every",5))
    for epoch in range(start_epoch,epochs):
        t0=time.time(); tr=_run_epoch(model,train_loader,criterion,device,optimizer,scaler,train_cfg.get("amp",True))
        with torch.no_grad(): va=_run_epoch(model,val_loader,criterion,device,None,None,train_cfg.get("amp",True))
        scheduler.step(); row={"epoch":epoch,"lr":optimizer.param_groups[0]["lr"],"seconds":time.time()-t0,"train":tr,"val":va}; history.append(row)
        print(f"epoch {epoch+1}/{epochs} train={tr['loss']:.4f} val={va['loss']:.4f} lr={row['lr']:.6g}")
        save_checkpoint(checkpoint_dir/"last.pt",model,optimizer,scheduler,epoch,best_val,model_cfg,{"history":history})
        if va["loss"]<best_val:
            best_val=va["loss"]; save_checkpoint(checkpoint_dir/"best.pt",model,optimizer,scheduler,epoch,best_val,model_cfg,{"history":history})
        if (epoch+1)%save_every==0: save_checkpoint(checkpoint_dir/f"epoch_{epoch+1:03d}.pt",model,optimizer,scheduler,epoch,best_val,model_cfg,{"history":history})
    return {"device":str(device),"epochs_completed":max(0,epochs-start_epoch),"best_val_loss":best_val,"checkpoint_dir":str(checkpoint_dir),"history":history}
