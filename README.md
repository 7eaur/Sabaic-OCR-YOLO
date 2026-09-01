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
- لا يتم اختلاق نتائج أو دقة؛ ملفات النتائج لا تُملأ إلا من تشغيل فعلي.

## نطاق النسخة الأولى

- 29 حرف Old South Arabian.
- الرمز U+10A7D ضمن الـclasses لأنه يُستخدم في المادة النصية كعلامة فصل/رقم واحد حسب السياق، ويساعد في بناء الكلمات.
- إجمالي classes في v1: **30**.
- U+10A7E وU+10A7F خارج نطاق v1، ويمكن إضافتهما لاحقًا إذا ظهرا بشكل كافٍ في البيانات الحقيقية.

## المكونات

```text
Sabaic-OCR-YOLO/
├── config/               # classes/model/train configuration
├── data/                 # local datasets (not committed by default)
├── docs/                 # architecture, labeling, experiment protocol
├── sabaic_ocr/
│   ├── data/             # dataset + augmentation + synthetic generation
│   ├── model/            # custom YOLO architecture/loss/decode/NMS
│   ├── ocr/              # reading order + Unicode reconstruction
│   └── metrics/          # detection/OCR metrics
├── scripts/              # train, finetune, infer, validate labels
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
4. جمع ووسم **200+ صورة حقيقية للتدريب** + validation/test حقيقية مستقلة.
5. Fine-tuning من synthetic checkpoint على Real Train فقط.
6. OCR post-processing: ترتيب الأسطر والحروف وتحويل Class IDs إلى Unicode.
7. Evaluation: Precision, Recall, mAP, Character Accuracy/CER, Word Accuracy/WER.
8. Inference pipeline ثم APK.
9. Report + PPT + final package.

## مهم

ملف الخط `NotoSansOldSouthArabian-Regular.ttf` لا يتم تضمينه تلقائيًا. يوضع محليًا في `assets/fonts/` عند مرحلة توليد البيانات، مع الالتزام بترخيص الخط.

## حالة المشروع

قيد البناء مرحلةً بمرحلة. لا توجد نتائج دقة معلنة حتى يتم إجراء تدريب واختبار فعليين.
