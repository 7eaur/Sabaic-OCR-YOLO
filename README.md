# Sabaic OCR YOLO

نظام OCR مخصص للخط السبئي (Old South Arabian / Musnad) مبني حول كاشف YOLO مكتوب داخل المشروع من الصفر باستخدام primitives في PyTorch، بدون Ultralytics أو نماذج YOLO جاهزة أو weights خارجية.

## هدف المشروع

تحويل صورة نقش سبئي إلى نص Unicode رقمي عبر المسار التالي:

`Image -> preprocessing -> YOLO character detection/classification -> reading order -> Unicode mapping -> OCR text`

## القيود الأكاديمية المعتمدة

- YOLO جزء أساسي وإجباري من مرحلة Object Detection.
- لا نستخدم Ultralytics أو أي implementation جاهز لـYOLO.
- لا نستخدم pretrained weights خارجية.
- ندرّب النموذج أولًا على Synthetic Data مولدة داخل المشروع؛ checkpoint الناتج هو نقطة البداية للـFine-tuning.
- Fine-tuning النهائي يتم على **200+ صورة حقيقية labeled للتدريب**، ولا تُحسب الصور الاصطناعية ضمن هذا العدد.
- نحتفظ بصور Real Validation وReal Test منفصلة عن صور الـ200+ المستخدمة في Fine-tuning.
- كل حرف ظاهر في الصورة الحقيقية المستخدمة للتدريب يجب أن يملك Bounding Box وClass ID صحيحًا.
- جميع مراحل التدريب والاستدلال Python ومتوافقة مع Google Colab وGPU وcheckpoint/resume.
- لا يتم اختلاق نتائج أو دقة؛ النتائج المعلنة في المستودع مأخوذة من تشغيل فعلي.

## نطاق النسخة الأولى

- 29 حرف Old South Arabian.
- الرمز U+10A7D ضمن الـclasses في النطاق الحالي.
- إجمالي classes: **30**.
- اتجاه إعادة البناء الأساسي في v1 هو RTL.

## المكونات

```text
Sabaic-OCR-YOLO/
├── config/               # classes/model/train configuration
├── data/                 # local datasets (not committed by default)
├── docs/                 # architecture, labeling, experiment protocol/results
├── sabaic_ocr/
│   ├── data/             # dataset + augmentation + synthetic generation
│   ├── model/            # custom YOLO architecture/loss/decode/NMS
│   ├── ocr/              # reading order + Unicode reconstruction
│   └── metrics/          # detection/OCR metrics
├── scripts/              # train, diagnose, evaluate, threshold sweep, finetune
├── notebooks/            # Google Colab entry points
├── tests/                # unit/integration tests
├── checkpoints/          # generated locally / Drive
├── outputs/              # generated locally
├── app/                  # APK stage
├── report/               # report stage
└── presentation/         # PPT stage
```

## مراحل المشروع

1. تثبيت نطاق الحروف والـUnicode والـarchitecture وسياسة الـlabeling.
2. بناء مولد Synthetic Data داخلي مع YOLO labels.
3. تدريب YOLO من الصفر على Synthetic Data وحفظ `best.pt` و`last.pt`.
4. تشخيص نتيجة v1 وإصلاح مشكلة تصنيف الحروف في Synthetic v2.
5. ضبط confidence التشغيلي للـOCR على Synthetic Test.
6. جمع ووسم **200+ صورة حقيقية للتدريب** + validation/test حقيقية مستقلة.
7. Fine-tuning من synthetic v2 checkpoint على Real Train فقط.
8. OCR post-processing: ترتيب الأسطر والحروف وتحويل Class IDs إلى Unicode.
9. Evaluation: Precision, Recall, mAP, Character Accuracy/CER, Word Accuracy/WER.
10. Inference pipeline ثم APK.
11. Report + PPT + final package.

## حالة التنفيذ الحالية

مرحلة **Synthetic pretraining v2** اكتملت وتم تقييمها فعليًا على Synthetic Test مستقل من 100 صورة.

### البيانات الاصطناعية

- Train: **5,000 صورة**.
- Validation: **500 صورة**.
- Test: **100 صورة / 1,289 حرفًا مرجعيًا**.
- جميع الـ30 class ممثلة، وتم اجتياز بوابة مراجعة الصور/labels/transcripts وعدم تسرب exact duplicates بين الـsplits.
- anchors محسوبة من البيانات، Mean Best IoU حوالي **0.87** وAnchor Recall@0.50 حوالي **1.0**.

### Synthetic v1

تم تدريب baseline أول لمدة 100 Epoch. التقييم كشف أن localization كان أفضل بكثير من classification؛ Character match accuracy كانت حوالي **20.56%** وCER حوالي **148.18%**، لذلك لم ننتقل إلى Fine-tuning الحقيقي مباشرة.

### Synthetic v2

بناءً على التشخيص تم:

- استبدال BCE classification بـCategorical Cross Entropy.
- إضافة class balancing.
- إعادة تهيئة classification logits فقط مع الحفاظ على box/objectness من v1.
- توحيد decode مع Softmax.
- اعتماد class-agnostic NMS لتقليل الاكتشافات المكررة.
- فصل confidence الخاص بحساب mAP عن confidence التشغيلي للـOCR.
- إضافة تدريب Resume-Safe يحفظ `last.pt` بعد كل Epoch على Google Drive.

اكتمل v2 لمدة **30/30 Epoch**. على Synthetic Test:

| Metric | النتيجة |
|---|---:|
| mAP50 | **0.9523** |
| mAP50-95 | **0.8634** |
| Recall@IoU50 | **0.9635** |

بعد Threshold Sweep، أفضل قيمة مجرّبة للـSynthetic OCR كانت `confidence=0.80`:

| OCR Metric | النتيجة |
|---|---:|
| CER | **0.0551** |
| WER | **0.2685** |
| Character match accuracy | **≈0.9604** |
| Insertions | **20** |
| Deletions | **8** |
| Substitutions | **43** |

هذه **نتائج Synthetic Test فقط وليست الدقة النهائية على الصور الحقيقية**. سيتم إعادة اختيار threshold على Real Validation بعد Fine-tuning، ثم إجراء التقييم النهائي على Real Test مستقل.

## التوثيق العربي الكامل للمرحلة الحالية

لشرح ما تم أمام الدكتور، بما في ذلك v1، سبب فشله، طريقة التشخيص، لماذا اخترنا كل تعديل في v2، وجداول النتائج والـThreshold Sweep:

**[`docs/07_synthetic_v2_results_ar.md`](docs/07_synthetic_v2_results_ar.md)**

وتوجد وثائق المراحل السابقة في `docs/01` إلى `docs/06`.

## المرحلة التالية

الانتقال إلى **Real Dataset**:

- تجهيز **200+ صورة حقيقية labeled للتدريب** على الأقل.
- تجهيز Real Validation وReal Test منفصلين.
- Bounding Box وClass ID صحيح لكل حرف ظاهر.
- مراجعة جودة الوسوم.
- Fine-tuning من `checkpoints/synthetic_v2/best.pt`.
- ضبط confidence/NMS باستخدام Real Validation فقط.
- تقييم نهائي على Real Test باستخدام mAP/CER/WER.

## مهم

ملف الخط `NotoSansOldSouthArabian-Regular.ttf` لا يتم تضمينه تلقائيًا في المستودع. يتم الاحتفاظ به محليًا/في Drive الخاص أثناء توليد البيانات، مع الالتزام بترخيصه.

الـcheckpoints الكبيرة وDataset archive ونتائج التشغيل المخزنة في Google Drive لا تُرفع تلقائيًا إلى GitHub.
