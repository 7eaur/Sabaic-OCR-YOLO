# Stage Gate — Synthetic Data -> Detector Pretraining

This document records the verified state of the synthetic-data stage before full YOLO pretraining. It is an audit record, **not** a model-accuracy report.

## Decision

**PASS — ready for full synthetic detector pretraining.**

The stage was re-audited after generation completed. The training code is allowed to start only when the configured minimum counts and label checks pass.

## Dataset actually reviewed

| Split | Images | Character boxes | All 30 classes present | Label/image errors | Empty labels/transcripts |
|---|---:|---:|---|---:|---:|
| Train | 5,000 | 65,513 | Yes | 0 | 0 |
| Validation | 500 | 6,580 | Yes | 0 | 0 |
| Synthetic test | 100 | 1,289 | Yes | 0 | 0 |

Additional checks:

- Exact image duplicates across train/validation/test: **0**.
- Bounding boxes were checked for finite normalized values and for staying inside image bounds.
- Corrupt images: **0**.
- Missing label files: **0**.
- Missing/empty transcripts: **0**.
- A random visual preview batch was inspected; boxes aligned with the rendered glyphs.

The synthetic sequences are detector-training material. Randomly generated sequences are **not claimed to be historically valid Sabaic text** unless a separately verified corpus is supplied.

## Final anchors from the reviewed Train split

The 9 priors were fitted from all **65,513** Train bounding boxes using IoU-distance k-means implemented in this repository.

```text
Scale 1: [20,38] [26,39] [29,50]
Scale 2: [35,50] [40,59] [37,73]
Scale 3: [48,68] [51,80] [61,81]
```

Anchor coverage:

- Mean best IoU: **0.8750**
- Anchor recall at IoU >= 0.50: **1.0000**
- Anchor recall at IoU >= 0.70: **0.9878**

These values are now stored in `config/model.json`. If the generator geometry changes materially, anchors must be fitted again.

## Model/loss training preflight

A deliberately tiny one-image overfit test was run only to verify that the custom model, target assignment, loss and optimizer can learn together.

- Ground-truth boxes: 3
- Steps: 80
- Initial total loss: **9.3006**
- Final total loss: **1.8248**
- Final / initial ratio: **0.1962**
- Required ratio <= 0.55: **PASS**

This result **must not be reported as OCR/detection accuracy**. It is only a plumbing sanity test.

## Code checks completed

- Unit tests: **9 passed**.
- Python compilation check: passed.
- Font validation: all configured 30 glyph classes render with the supplied Noto Sans Old South Arabian font.
- Synthetic generation supports safe resume/overwrite behavior and refuses inconsistent partial splits.
- Training entry point refuses incomplete datasets or missing classes.
- Real fine-tuning remains separately locked to at least **200 real labeled Train images**.

## Important fixes made during review

1. A partially generated dataset was detected and blocked instead of being treated as ready.
2. Label validation was strengthened to reject non-finite and out-of-image boxes and to detect corrupt/empty samples.
3. Cross-split exact-duplicate leakage checks were added.
4. OCR word evaluation was corrected so a physical line break is not automatically treated as a word boundary; the encoded U+10A7D separator is the word-boundary signal used by the evaluator.
5. Checkpoint resume behavior was hardened for long Colab runs.

## Next stage

Full synthetic pretraining is the next stage. It should run on a Google Colab GPU using `config/train_synthetic.json`, producing actual `best.pt` and `last.pt` checkpoints. No final accuracy is claimed until that training and evaluation are actually executed.
