# Runbook — من الصفر إلى التقييم

> النموذج والـloss والـNMS ومولد الصور مكتوبة داخل هذا المستودع. لا يوجد Ultralytics/TRDG أو pretrained weights خارجية.

## 0) Setup

```bash
pip install -r requirements.txt
pip install -e . --no-deps
python scripts/check_environment.py
pytest -q
```

## 1) Font

ضع محليًا:

```text
assets/fonts/NotoSansOldSouthArabian-Regular.ttf
```

ثم افحص دعم جميع الـclasses:

```bash
python scripts/validate_font.py
```

## 2) Synthetic generation

تجربة صغيرة:

```bash
python scripts/generate_synthetic.py --train 500 --val 100 --test 50
```

المجموعة المقترحة للتدريب الكامل على Colab/GPU:

```bash
python scripts/generate_synthetic.py --train 10000 --val 1500 --test 500
```

ثم:

```bash
python scripts/validate_labels.py \
  --images data/synthetic/images/train \
  --labels data/synthetic/labels/train \
  --preview-dir outputs/synthetic_previews

python scripts/audit_synthetic.py \
  --images data/synthetic/images/train \
  --labels data/synthetic/labels/train \
  --output outputs/synthetic_audit.json
```

## 3) Anchors

```bash
python scripts/fit_anchors.py \
  --images data/synthetic/images/train \
  --labels data/synthetic/labels/train
```

القيم الحالية في `config/model.json` حُسبت من دفعة مراجعة فعلية تحتوي 30,234 Bounding Boxes. عند تغيير طريقة التوليد بشكل كبير يعاد حسابها.

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

> لا تعتبر smoke tests الموجودة في وثائق المراجعة نتائج دقة نهائية. النتائج النهائية لا تسجل إلا بعد التدريب الكامل.

## 5) Real data

المطلوب للـfine-tuning:

```text
data/real/images/train     >= 200 real images
data/real/labels/train     label per image
```

هذه الـ200+ صورة يجب أن تكون **حقيقية وليست Synthetic**، وكل حرف ظاهر مستخدم في التدريب يأخذ Bounding Box وClass ID صحيحًا. الصورة الواحدة يمكن أن تحتوي عددًا قليلًا من الحروف.

مع Validation/Test حقيقيين منفصلين، ويفضل إبقاء كل crops العائدة لنفس النقش في split واحد لمنع Data Leakage.

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

يحفظ:

- Precision / Recall / mAP50 / mAP50:95.
- Correct/Wrong characters + CER + Character match accuracy.
- Correct/Wrong words + WER + Word match accuracy.

في:

```text
outputs/evaluation/test_metrics.json
```

## 8) Inference

```bash
python scripts/infer.py \
  --checkpoint checkpoints/real_finetune/best.pt \
  --image path/to/new_image.jpg
```

المخرجات: صورة Bounding Boxes + Class IDs + Confidence + JSON + النص السبئي Unicode.
