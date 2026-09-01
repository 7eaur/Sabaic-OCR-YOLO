from __future__ import annotations

from pathlib import Path
import torch


def save_checkpoint(path, model, optimizer, scheduler, epoch: int, best_metric: float, model_config: dict, extra: dict|None=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({"epoch":int(epoch),"best_metric":float(best_metric),"model_state":model.state_dict(),"optimizer_state":optimizer.state_dict() if optimizer is not None else None,"scheduler_state":scheduler.state_dict() if scheduler is not None else None,"model_config":model_config,"extra":extra or {}},path)


def load_checkpoint(path, model, optimizer=None, scheduler=None, device="cpu", strict: bool=True):
    payload=torch.load(path,map_location=device,weights_only=False); model.load_state_dict(payload["model_state"],strict=strict)
    if optimizer is not None and payload.get("optimizer_state") is not None: optimizer.load_state_dict(payload["optimizer_state"])
    if scheduler is not None and payload.get("scheduler_state") is not None: scheduler.load_state_dict(payload["scheduler_state"])
    return payload
