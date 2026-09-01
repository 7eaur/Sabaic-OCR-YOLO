# المرحلة 1 — Architecture ونطاق النظام

## القرار المعماري

المشروع يستخدم **YOLO مكتوب داخل المستودع** بدل استخدام Ultralytics أو نموذج Detection جاهز.

```text
Image
  ↓
Letterbox + photometric preprocessing
  ↓
Custom CNN backbone
  ↓
CSP/FPN/PAN-style feature fusion
  ↓
3 YOLO detection heads (strides 8, 16, 32)
  ↓
Bounding box + Objectness + Character Class
  ↓
Custom decode + class-aware NMS
  ↓
Line clustering + RTL ordering
  ↓
Class ID → Old South Arabian Unicode
  ↓
OCR text
```

## Detection scales

ثلاثة رؤوس كشف عند strides 8 و16 و32 لدعم الحروف الصغيرة والمتوسطة والكبيرة. كل رأس يستخدم 3 Anchors. توجد أداة داخلية `scripts/fit_anchors.py` لحساب Anchors من الداتا بدون sklearn.

## Classes

النسخة الأولى تستخدم 30 Class: 29 حرفًا U+10A60..U+10A7C، إضافة إلى U+10A7D بالاسم الرسمي `OSA_NUMBER_ONE`. لا نغير الاسم الرسمي إلى word separator داخل الـmapping حتى لا نخلط بين Unicode والوظيفة السياقية.

## التدريب بدون Model جاهز

1. Synthetic pretraining: تهيئة عشوائية ثم تدريب على صور مولدة داخل المشروع وحفظ `checkpoints/synthetic/best.pt`.
2. Real fine-tuning: تحميل checkpoint السابق ثم التدريب على **200+ صورة حقيقية labeled في train split**. Validation/Test الحقيقيان منفصلان.

بهذا لا توجد pretrained weights خارجية، لكن Fine-tuning لا يبدأ من الصفر.

## Loss وNMS

- CIoU للـboxes.
- BCE للـobjectness.
- BCE للـclasses.
- Anchor assignment باستخدام width/height IoU.
- NMS مكتوب بـPyTorch داخل المشروع، بدون `torchvision.ops.nms`.

## اتجاه القراءة

v1 يعيد بناء الأسطر RTL. Boustrophedon لا يتم تخمينه تلقائيًا؛ يضاف لاحقًا كتوسعة إذا كانت البيانات الموثقة تتطلبه.
