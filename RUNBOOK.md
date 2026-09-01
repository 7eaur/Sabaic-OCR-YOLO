# Runbook — من الصفر إلى التقييم

> النموذج والـloss والـNMS ومولد الصور مكتوبة داخل هذا المستودع. لا يوجد Ultralytics/TRDG أو pretrained weights خارجية.

## 0) Setup

```bash
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/check_environment.py
python -m unittest discover -s tests -v
```

## 1) Font

ضع محليًا:

```text
assets/fonts/NotoSansOldSouthArabian-Regular.ttf
```

## 2) Synthetic generation

تجربة صغيرة:

```bash
python scripts/generate_synthetic.py --train 500 --val 100
```

المجموعة الكاملة المقترحة كبداية:

```bash
python scripts/generate_synthetic.py --train 10000 --val 1500
```

ثم:

```bash
python scripts/validate_labels.py \
  --images data/synthetic/images/train \
  --labels data/synthetic/labels/train \
  --preview-dir outputs/synthetic_previews
```

## 3) Fit anchors

```bash
python scripts/fit_anchors.py \
  --images data/synthetic/images/train \
  --labels data/synthetic/labels/train
```

راجع القيم الناتجة ثم ضعها في `config/model.json`.

## 4) Synthetic pretraining

```bash
python scripts/train_synthetic.py
```

المخرجات:

```text
checkpoints/synthetic/best.pt
checkpoints/synthetic/last.pt
```

يمكن Resume عبر وضع مسار `last.pt` في `config/train_synthetic.json -> resume`.

## 5) Real data

المطلوب للـfine-tuning:

```text
data/real/images/train     >= 200 real images
data/real/labels/train     label per image
```

مع Validation/Test حقيقيين منفصلين.

تحقق قبل التدريب:

```bash
python scripts/validate_labels.py \
  --images data/real/images/train \
  --labels data/real/labels/train \
  --preview-dir outputs/real_label_previews
```

## 6) Fine-tuning

```bash
python scripts/finetune_real.py
```

السكربت يرفض التشغيل إذا:

- صور Real Train أقل من 200.
- Labels ناقصة/فاسدة.
- Synthetic pretrained checkpoint غير موجود.

## 7) Final evaluation

```bash
python scripts/evaluate.py \
  --checkpoint checkpoints/real_finetune/best.pt
```

يحفظ Detection + Character OCR + Word OCR metrics في:

```text
outputs/evaluation/test_metrics.json
```

## 8) Inference

```bash
python scripts/infer.py \
  --checkpoint checkpoints/real_finetune/best.pt \
  --image path/to/new_image.jpg
```

المخرجات: صورة Bounding Boxes + JSON + النص السبئي Unicode.
