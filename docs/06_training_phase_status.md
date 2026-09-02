# Detector Training Phase — Historical Preflight Status

> هذه الوثيقة توثّق **مرحلة ما قبل التدريب الكامل** تاريخيًا. حالة المشروع الحالية والنتائج الفعلية بعد v1 وv2 موثقة في:
>
> **[`docs/07_synthetic_v2_results_ar.md`](07_synthetic_v2_results_ar.md)**

## ما الذي كان يتم التحقق منه هنا؟

قبل صرف وقت GPU على التدريب الكامل، تم تشغيل المسار الكامل على بيانات smoke صغيرة للتأكد من أن:

```text
DataLoader -> custom YOLO -> custom loss -> optimizer -> validation -> checkpoint -> resume
```

يعمل بدون أخطاء.

أول smoke epoch:

```text
epoch 1/1 train=9.9365 val=9.4785
```

ثم تم التحقق من أن `last.pt` يمكن تحميله والاستكمال بدل إعادة البداية:

```text
epoch 2/2 train=9.5263 val=10.5408
```

هذه الأرقام **اختبار pipeline/resume فقط** وليست دقة للمشروع.

## CI والاختبارات

تم تجهيز GitHub Actions لفحص compilation وتشغيل اختبارات المشروع، كما تم تنفيذ اختبارات Unit/Integration قبل التدريب الطويل.

## الهدف الذي كان مخططًا في هذه المرحلة

كان الهدف الأصلي للتدريب الاصطناعي الأول:

- 5,000 Train images.
- 500 Validation images.
- 640×640 input.
- 30 classes.
- 100 epochs.
- batch size 16.
- AMP على CUDA.
- حفظ `best.pt` و`last.pt` في Google Drive.

## ما حدث بعد هذه الوثيقة؟

تم لاحقًا تنفيذ التدريب الكامل فعلًا على GPU:

1. اكتمل Synthetic v1 لمدة 100/100 Epoch.
2. تقييم v1 كشف انهيار classification رغم أن localization كان أفضل.
3. أضيف تشخيص يفصل localization عن classification.
4. تم بناء Synthetic v2 مع Cross Entropy وclass balancing وإعادة تهيئة classification logits فقط.
5. اكتمل v2 لمدة 30/30 Epoch.
6. حقق Synthetic Test تقريبًا `mAP50=0.9523` و`mAP50-95=0.8634`.
7. تم تنفيذ Threshold Sweep وانخفض CER المقاس إلى `0.0551` عند confidence `0.80` على Synthetic Test.

لذلك أي عبارة قديمة من نوع "لم يتم التدريب الكامل بعد" تعتبر **متجاوزة زمنيًا** ولا تمثل الحالة الحالية. المرجع الحالي هو الوثيقة رقم 07 وREADME.
