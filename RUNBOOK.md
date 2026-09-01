# Runbook — من الصفر إلى التقييم

> النموذج والـloss والـNMS ومولد الصور مكتوبة داخل هذا المستودع. لا يوجد Ultralytics/TRDG أو pretrained weights خارجية.

## 0) Setup

```bash
pip install -r requirements.txt
pip install -e . --no-deps --no-build-isolation
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

تجربة صغيرة للفحص فقط:

```bash
python scripts/generate_synthetic.py --train 500 --val 100 --test 50 --overwrite
```

المجموعة المعتمدة للـbaseline الحالي:

```bash
python scripts/generate_synthetic.py --train 5000 --val 500 --test 100 --overwrite
```

إذا انقطع التوليد، لا تعِد الكتابة فوق البيانات الموجودة؛ استخدم:

```bash
python scripts/generate_synthetic.py --train 5000 --val 500 --test 100 --resume
```

ثم نفّذ بوابة المراجعة الكاملة:

```bash
python scripts/review_synthetic_stage.py \
  --require-train 5000 \
  --require-val 500 \
  --require-test 100
```

وعند الحاجة لمعاينة الصناديق بصريًا:

```bash
python scripts/validate_labels.py \
  --images data/synthetic/images/train \
  --labels data/synthetic/labels/train \
  --preview-dir outputs/synthetic_previews
```

## 3) Anchors

```bash
python scripts/fit_anchors.py \
  --images data/synthetic/images/train \
  --labels data/synthetic/labels/train
```

القيم الحالية في `config/model.json` حُسبت من مجموعة Train المراجعة كاملة: **5,000 صورة و65,513 Bounding Boxes**.

```text
Scale 1: [20,38] [26,39] [29,50]
Scale 2: [35,50] [40,59] [37,73]
Scale 3: [48,68] [51,80] [61,81]
```

جودة التغطية المقاسة: mean best IoU = 0.8750، recall@0.50 = 1.0000، recall@0.70 = 0.9878. عند تغيير هندسة التوليد بشكل كبير يجب حساب الـanchors من جديد.

## 4) Training preflight ثم Synthetic pretraining

قبل التدريب الطويل:

```bash
python scripts/sanity_overfit.py
```

هذا اختبار plumbing فقط وليس نتيجة دقة.

ثم على Google Colab مع GPU:

```bash
python scripts/train_synthetic.py
```

المخرجات:

```text
checkpoints/synthetic/best.pt
checkpoints/synthetic/last.pt
```

يمكن Resume عبر وضع مسار `last.pt` في `config/train_synthetic.json -> resume`.

> لا تعتبر smoke/preflight tests نتائج دقة نهائية. النتائج النهائية لا تسجل إلا بعد التدريب الكامل والتقييم الفعلي.

## 5) Real data

المطلوب للـfine-tuning:

```text
data/real/images/train     >= 200 real images
data/real/labels/train     label per image
```

هذه الـ200+ صورة يجب أن تكون **حقيقية وليست Synthetic**، وكل حرف ظاهر مستخدم في التدريب يأخذ Bounding Box وClass ID صحيحًا. الصورة الواحدة يمكن أن تحتوي عددًا قليلًا من الحروف.

مع Validation/Test حقيقيين منفصلين، ويفضل إبقاء كل crops العائدة لنفس النقش في split واحد لمنع Data Leakage.

تحقق قبل التدريب، بما في ذلك منع تسرّب الصور بين الـsplits ووجود Ground Truth للاختبار:

```bash
python scripts/audit_real_dataset.py --min-train 200 --min-val 20 --min-test 20

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
- Labels ناقصة/فاسدة أو صور تالفة/Labels فارغة.
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
