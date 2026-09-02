from __future__ import annotations

import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from sabaic_ocr.data.augment import PhotometricAugment
from sabaic_ocr.data.dataset import YoloCharacterDataset, yolo_collate
from sabaic_ocr.model.loss import YoloLoss
from sabaic_ocr.model.yolo import SabaicYOLO
from sabaic_ocr.training.checkpoint import load_checkpoint, save_checkpoint


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_cfg: dict) -> SabaicYOLO:
    return SabaicYOLO(
        num_classes=model_cfg["num_classes"],
        width_mult=model_cfg.get("width_mult", 0.5),
        depth_mult=model_cfg.get("depth_mult", 0.5),
    )


def build_loader(
    images_dir: str,
    labels_dir: str,
    model_cfg: dict,
    batch_size: int,
    num_workers: int,
    augment: bool,
    shuffle: bool,
) -> DataLoader:
    aug = PhotometricAugment() if augment else None
    ds = YoloCharacterDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        num_classes=model_cfg["num_classes"],
        image_size=model_cfg["image_size"],
        augment=aug,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=yolo_collate,
        drop_last=False,
    )


def cosine_lambda(epoch: int, total_epochs: int, warmup_epochs: int) -> float:
    if epoch < warmup_epochs:
        return max(0.05, (epoch + 1) / max(1, warmup_epochs))
    progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
    return 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))


def compute_class_weights(labels_dir: str | Path, num_classes: int) -> list[float]:
    """Inverse-sqrt frequency weights, normalized to mean 1 and softly clipped.

    This is intentionally milder than pure inverse-frequency weighting: the
    separator is naturally more common than an individual letter, but it should
    not dominate the classification objective as it did in the first baseline.
    """
    counts = np.zeros(int(num_classes), dtype=np.int64)
    for path in Path(labels_dir).glob("*.txt"):
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if 0 <= cls_id < num_classes:
                counts[cls_id] += 1
    if (counts == 0).any():
        missing = np.where(counts == 0)[0].tolist()
        raise RuntimeError(f"Cannot class-balance training: classes with zero labels: {missing}")
    weights = 1.0 / np.sqrt(counts.astype(np.float64))
    weights /= weights.mean()
    weights = np.clip(weights, 0.5, 2.5)
    weights /= weights.mean()
    print("class_counts:", counts.tolist())
    print("class_weights:", [round(float(x), 4) for x in weights.tolist()])
    return [float(x) for x in weights.tolist()]


def reset_classification_logits(model: SabaicYOLO) -> None:
    """Reinitialize only class-output channels while preserving box/objectness.

    Useful for corrective synthetic training after a checkpoint has learned good
    localization but a collapsed classifier. The detector architecture remains
    unchanged and no external weights are introduced.
    """
    num_classes = model.num_classes
    with torch.no_grad():
        for head in (model.head_s8, model.head_s16, model.head_s32):
            for anchor_idx in range(model.num_anchors):
                base = anchor_idx * (5 + num_classes)
                start = base + 5
                end = start + num_classes
                torch.nn.init.normal_(head.weight[start:end], mean=0.0, std=0.01)
                if head.bias is not None:
                    head.bias[start:end].zero_()


def _run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    scaler=None,
    amp: bool = True,
) -> dict:
    train = optimizer is not None
    model.train(train)

    total = box = obj = cls = 0.0
    positives = 0
    batches = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = [t.to(device, non_blocking=True) for t in targets]

        if train:
            optimizer.zero_grad(set_to_none=True)

        use_amp = bool(amp and device.type == "cuda")
        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                predictions = model(images)
                loss = criterion(predictions, targets)

            if train:
                if scaler is not None and use_amp:
                    scaler.scale(loss.total).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.total.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

        total += float(loss.total.detach().item())
        box += float(loss.box.item())
        obj += float(loss.obj.item())
        cls += float(loss.cls.item())
        positives += loss.positives
        batches += 1

    denom = max(1, batches)
    return {
        "loss": total / denom,
        "box_loss": box / denom,
        "obj_loss": obj / denom,
        "cls_loss": cls / denom,
        "positives": positives,
    }


def train_detector(
    train_cfg: dict,
    model_cfg: dict,
    init_checkpoint: str | None = None,
    require_min_train_images: int | None = None,
) -> dict:
    set_seed(int(train_cfg.get("seed", 42)))

    if require_min_train_images is not None:
        from sabaic_ocr.data.dataset import list_images
        count = len(list_images(train_cfg["images_dir"]))
        if count < require_min_train_images:
            raise RuntimeError(
                f"Fine-tuning requires at least {require_min_train_images} real train images; found {count}."
            )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_cfg).to(device)

    class_weights = None
    if bool(train_cfg.get("class_balance", False)):
        class_weights = compute_class_weights(
            train_cfg["labels_dir"], int(model_cfg["num_classes"])
        )

    criterion = YoloLoss(
        anchors=model_cfg["anchors"],
        num_classes=model_cfg["num_classes"],
        image_size=model_cfg["image_size"],
        box_weight=float(train_cfg.get("box_weight", 5.0)),
        obj_weight=float(train_cfg.get("obj_weight", 1.0)),
        cls_weight=float(train_cfg.get("cls_weight", 1.0)),
        noobj_weight=float(train_cfg.get("noobj_weight", 0.25)),
        cls_label_smoothing=float(train_cfg.get("cls_label_smoothing", 0.02)),
        class_weights=class_weights,
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        momentum=float(train_cfg.get("momentum", 0.937)),
        weight_decay=float(train_cfg.get("weight_decay", 5e-4)),
        nesterov=True,
    )

    epochs = int(train_cfg["epochs"])
    warmup_epochs = int(train_cfg.get("warmup_epochs", 3))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda e: cosine_lambda(e, epochs, warmup_epochs),
    )

    train_loader = build_loader(
        train_cfg["images_dir"],
        train_cfg["labels_dir"],
        model_cfg,
        int(train_cfg["batch_size"]),
        int(train_cfg.get("num_workers", 2)),
        bool(train_cfg.get("augmentation", True)),
        True,
    )
    val_loader = build_loader(
        train_cfg["val_images_dir"],
        train_cfg["val_labels_dir"],
        model_cfg,
        int(train_cfg["batch_size"]),
        int(train_cfg.get("num_workers", 2)),
        False,
        False,
    )

    start_epoch = 0
    best_val = float("inf")

    resume_path = train_cfg.get("resume") or ""
    if resume_path:
        payload = load_checkpoint(
            resume_path, model, optimizer=optimizer, scheduler=scheduler, device=device
        )
        start_epoch = int(payload.get("epoch", -1)) + 1
        best_val = float(payload.get("best_metric", best_val))
    elif init_checkpoint:
        payload = load_checkpoint(init_checkpoint, model, device=device)
        ckpt_classes = payload.get("model_config", {}).get("num_classes")
        if ckpt_classes is not None and int(ckpt_classes) != int(model_cfg["num_classes"]):
            raise RuntimeError("Checkpoint num_classes does not match current model config.")
        if bool(train_cfg.get("reset_classification_head", False)):
            reset_classification_logits(model)
            print("classification logits reset; box/objectness channels preserved")

    amp_enabled = bool(train_cfg.get("amp", True) and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled) if device.type == "cuda" else None

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history = []
    save_every = int(train_cfg.get("save_every", 5))

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        train_stats = _run_epoch(
            model, train_loader, criterion, device, optimizer, scaler, train_cfg.get("amp", True)
        )
        with torch.no_grad():
            val_stats = _run_epoch(
                model, val_loader, criterion, device, optimizer=None, scaler=None, amp=train_cfg.get("amp", True)
            )

        current_lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "lr": current_lr,
            "seconds": time.time() - t0,
            "train": train_stats,
            "val": val_stats,
        }
        history.append(row)
        print(
            f"epoch {epoch+1}/{epochs} "
            f"train={train_stats['loss']:.4f} val={val_stats['loss']:.4f} "
            f"box={val_stats['box_loss']:.4f} obj={val_stats['obj_loss']:.4f} "
            f"cls={val_stats['cls_loss']:.4f} lr={row['lr']:.6g}"
        )

        scheduler.step()

        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            save_checkpoint(
                checkpoint_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                model_cfg,
                extra={"history": history, "train_config": train_cfg},
            )

        save_checkpoint(
            checkpoint_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            epoch,
            best_val,
            model_cfg,
            extra={"history": history, "train_config": train_cfg},
        )

        if (epoch + 1) % save_every == 0:
            save_checkpoint(
                checkpoint_dir / f"epoch_{epoch+1:03d}.pt",
                model,
                optimizer,
                scheduler,
                epoch,
                best_val,
                model_cfg,
                extra={"history": history, "train_config": train_cfg},
            )

    return {
        "device": str(device),
        "epochs_completed": max(0, epochs - start_epoch),
        "best_val_loss": best_val,
        "checkpoint_dir": str(checkpoint_dir),
        "history": history,
    }
