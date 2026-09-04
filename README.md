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
├── scripts/              # train, diagnose, evaluate, threshold sweep, remap, finetune
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
6. تنفيذ Real Pilot صغير بدون Fine-tuning لقياس الـDomain Gap.
7. جمع ووسم **200+ صورة حقيقية للتدريب** + validation/test حقيقية مستقلة.
8. Fine-tuning من synthetic v2 checkpoint على Real Train فقط.
9. OCR post-processing: ترتيب الأسطر والحروف وتحويل Class IDs إلى Unicode.
10. Evaluation: Precision, Recall, mAP, Character Accuracy/CER, Word Accuracy/WER.
11. Inference pipeline ثم APK.
12. Report + PPT + final package.

# حالة التنفيذ الحالية

## 1) Synthetic v1

تم تدريب baseline أول لمدة 100 Epoch. التقييم كشف أن localization كان أفضل بكثير من classification؛ Character match accuracy كانت حوالي **20.56%** وCER حوالي **148.18%**، لذلك لم ننتقل إلى Fine-tuning الحقيقي مباشرة.

## 2) Synthetic v2

بناءً على التشخيص تم:

- استبدال BCE classification بـCategorical Cross Entropy.
- إضافة class balancing.
- إعادة تهيئة classification logits فقط مع الحفاظ على box/objectness من v1.
- توحيد decode مع Softmax.
- اعتماد class-agnostic NMS لتقليل الاكتشافات المكررة.
- فصل confidence الخاص بحساب mAP عن confidence التشغيلي للـOCR.
- إضافة تدريب Resume-Safe يحفظ `last.pt` بعد كل Epoch على Google Drive.

اكتمل v2 لمدة **30/30 Epoch**. على Synthetic Test من 100 صورة / 1,289 حرفًا:

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

هذه **نتائج Synthetic Test فقط وليست الدقة النهائية على الصور الحقيقية**.

## 3) أول Real Pilot

تم بعد ذلك اختبار نفس checkpoint `synthetic_v2/best.pt` على **5 صور حقيقية تجريبية مصورة** لم تدخل في التدريب، وبها **17 Bounding Box** عبر 15 Class.

عند `confidence=0.50`:

| Metric | Real Pilot |
|---|---:|
| Predictions | 45 |
| Ground-truth boxes | 17 |
| Localization Precision | **15.56%** |
| Localization Recall | **41.18%** |
| Classification Accuracy on localized matches | **57.14%** |
| End-to-End Same-Class Recall | **23.53%** |
| Mean IoU of localized matches | **0.6293** |

هذا أكد وجود **Domain Gap** واضحة بين Synthetic والـReal.

### تشخيص Class 29

في الـPilot لم يوجد Class 29 داخل Ground Truth، لكن النموذج تنبأ به بكثرة على خطوط/حواف شبيهة بالرمز:

- عند conf=0.25: `21/64 = 32.81%` من predictions كانت Class 29.
- عند conf=0.50: `11/45 = 24.44%`.
- عند conf=0.80: `4/24 = 16.67%`.

وعند تجاهل Class 29 تشخيصيًا فقط عند conf=0.50:

```text
Predictions:             45 -> 34
Localization matches:     7 -> 7
Localization Precision: 15.56% -> 20.59%
Localization Recall:    41.18% -> 41.18%
```

إذن Class 29 مسؤول عن جزء واضح من الـFalse Positives لكنه ليس المشكلة الوحيدة. لا يتم حذف Class 29 من المشروع؛ العلاج الصحيح هو Real Fine-tuning وبيانات أكثر تنوعًا.

## 4) مشكلة Roboflow Class IDs وحلها

أول تصدير تجريبي احتوى 15 Class فقط، فقام Roboflow بإعادة ترقيمها داخليًا من 0 إلى 14 رغم أن أسماء الـClasses كانت أرقام المشروع مثل `00`, `01`, `03`, `05`, ...

تمت إضافة:

```text
scripts/remap_roboflow_labels.py
```

لاستعادة Project IDs الثابتة 0..29 تلقائيًا اعتمادًا على اسم الـClass، بدون تغيير الصور أو إحداثيات الـBounding Boxes.

# التوثيق العربي

- **[`docs/07_synthetic_v2_results_ar.md`](docs/07_synthetic_v2_results_ar.md)** — توثيق Synthetic v1/v2، التشخيص، التعديلات والنتائج.
- **[`docs/08_real_pilot_results_ar.md`](docs/08_real_pilot_results_ar.md)** — توثيق Real Pilot الأول، Domain Gap، Class 29 والنتائج بالأرقام.
- **[`docs/09_real_data_labeler_handoff_ar.md`](docs/09_real_data_labeler_handoff_ar.md)** — تعليمات مفصلة جاهزة للشخص الذي يجمع الصور ويعمل Annotation.
- **[`docs/02_labeling_protocol.md`](docs/02_labeling_protocol.md)** — البروتوكول الرسمي المختصر للـReal Labeling.

# المرحلة التالية

الآن لا يتم تدريب النموذج على صور الـPilot الخمس. تُحفظ كتجربة تشخيصية قبل الـFine-tuning.

العمل التالي هو:

- جمع ما يقارب 250–280 صورة واقعية حتى يبقى **Real Train 200+** بعد المراجعة والتقسيم.
- إعطاء الأولوية للصور التي تمثل المجال النهائي للتطبيق، خصوصًا النقوش/المسند الواقعي إذا كان هذا هو مجال التطبيق النهائي.
- عمل Bounding Box مستقل لكل رمز واضح.
- استخدام أسماء Classes في Roboflow من `00` إلى `29` حسب `config/classes.json`.
- إرسال أول 20 صورة موسومة للمراجعة قبل إكمال بقية الـDataset.
- تسجيل مصدر كل صورة وتجميع الصور العائدة لنفس النقش لمنع Data Leakage.
- بعد المراجعة: تقسيم Real Train/Validation/Test، ثم Fine-tuning من `checkpoints/synthetic_v2/best.pt`.
- اختيار confidence/NMS باستخدام Real Validation فقط، ثم التقييم النهائي على Real Test مستقل.

## مهم

ملف الخط `NotoSansOldSouthArabian-Regular.ttf` لا يتم تضمينه تلقائيًا في المستودع. يتم الاحتفاظ به محليًا/في Drive الخاص أثناء توليد البيانات، مع الالتزام بترخيصه.

الـcheckpoints الكبيرة وDataset archive ونتائج التشغيل المخزنة في Google Drive لا تُرفع تلقائيًا إلى GitHub.
