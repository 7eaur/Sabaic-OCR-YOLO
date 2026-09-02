#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sabaic_ocr.config import load_classes
from sabaic_ocr.data.dataset import YoloCharacterDataset, yolo_collate
from sabaic_ocr.metrics.ocr import evaluate_corpus
from sabaic_ocr.model.decode import postprocess_batch
from sabaic_ocr.ocr.postprocess import detections_to_text, tensor_detections_to_objects
from sabaic_ocr.training.engine import build_model


def parse_thresholds(text: str) -> list[float]:
    values = [float(x.strip()) for x in text.split(',') if x.strip()]
    if not values:
        raise ValueError('At least one threshold is required.')
    for value in values:
        if not 0.0 < value < 1.0:
            raise ValueError(f'Threshold must be between 0 and 1: {value}')
    return values


def main() -> None:
    p = argparse.ArgumentParser(description='Sweep OCR confidence thresholds on one fixed test set.')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--images', default='data/synthetic/images/test')
    p.add_argument('--labels', default='data/synthetic/labels/test')
    p.add_argument('--transcripts', default='data/synthetic/transcripts/test')
    p.add_argument('--classes', default='config/classes.json')
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--iou', type=float, default=0.45)
    p.add_argument(
        '--thresholds',
        default='0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60',
        help='Comma-separated OCR confidence thresholds.',
    )
    p.add_argument('--output', default='outputs/evaluation/ocr_threshold_sweep.json')
    args = p.parse_args()

    thresholds = parse_thresholds(args.thresholds)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = payload['model_config']
    classes = load_classes(args.classes)

    model = build_model(cfg).to(device)
    model.load_state_dict(payload['model_state'], strict=True)
    model.eval()

    ds = YoloCharacterDataset(
        args.images,
        args.labels,
        cfg['num_classes'],
        cfg['image_size'],
        None,
        True,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=yolo_collate,
    )

    id_to_char = {int(c['id']): c['char'] for c in classes['classes']}
    pairs_by_threshold: dict[float, list[tuple[str, str]]] = {t: [] for t in thresholds}

    with torch.no_grad():
        for images, _targets, metas in loader:
            raw = model(images.to(device))
            detections_by_threshold = {
                threshold: postprocess_batch(
                    raw,
                    cfg['anchors'],
                    cfg['num_classes'],
                    cfg['image_size'],
                    threshold,
                    args.iou,
                )
                for threshold in thresholds
            }

            for index, meta in enumerate(metas):
                gt_path = Path(args.transcripts) / f"{Path(meta['image_path']).stem}.txt"
                if not gt_path.exists():
                    raise FileNotFoundError(f'Missing test transcript: {gt_path}')
                reference = gt_path.read_text(encoding='utf-8').strip()

                for threshold in thresholds:
                    det = detections_by_threshold[threshold][index].cpu()
                    pred_text = detections_to_text(
                        tensor_detections_to_objects(det), id_to_char, 'rtl'
                    )
                    pairs_by_threshold[threshold].append((reference, pred_text))

    rows = []
    for threshold in thresholds:
        ocr = evaluate_corpus(pairs_by_threshold[threshold])
        character = ocr['character']
        word = ocr['word']
        row = {
            'threshold': threshold,
            'character': character,
            'word': word,
        }
        rows.append(row)
        print(
            f"conf={threshold:.2f} "
            f"CER={character['cer']:.4f} "
            f"match_acc={character['match_accuracy']:.4f} "
            f"pred={character['predicted_count']} "
            f"ins={character['insertions']} "
            f"del={character['deletions']} "
            f"sub={character['substitutions']} "
            f"WER={word['wer']:.4f}"
        )

    best = min(
        rows,
        key=lambda row: (
            row['character']['cer'],
            row['word']['wer'],
            row['character']['insertions'],
        ),
    )

    report = {
        'checkpoint': args.checkpoint,
        'test_images': len(ds),
        'nms_iou_threshold': args.iou,
        'thresholds': thresholds,
        'results': rows,
        'best_by_cer': best,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\nBEST')
    print(
        f"threshold={best['threshold']:.2f} "
        f"CER={best['character']['cer']:.4f} "
        f"WER={best['word']['wer']:.4f} "
        f"insertions={best['character']['insertions']} "
        f"deletions={best['character']['deletions']} "
        f"substitutions={best['character']['substitutions']}"
    )
    print('Saved:', out)


if __name__ == '__main__':
    main()
