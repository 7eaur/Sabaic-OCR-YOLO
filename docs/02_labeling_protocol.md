# بروتوكول الـLabelling للصور الحقيقية

هذه الوثيقة هي القاعدة الرسمية المختصرة للـReal Dataset. التعليمات التفصيلية الجاهزة للتسليم للشخص الذي يقوم بالوسم موجودة في:

**[`docs/09_real_data_labeler_handoff_ar.md`](09_real_data_labeler_handoff_ar.md)**

وتوثيق أول Real Pilot وسبب التشديد على Class 29 موجود في:

**[`docs/08_real_pilot_results_ar.md`](08_real_pilot_results_ar.md)**

---

## شرط Fine-tuning

`data/real/images/train` يجب أن يحتوي **200 صورة حقيقية على الأقل** بعد المراجعة. الصور الاصطناعية لا تدخل في هذا العدد.

يجب أن تكون هناك صور Real Validation وReal Test مستقلة عن Train. يفضّل جمع 250–280 صورة تقريبًا حتى يبقى Train نفسه 200+ بعد الاستبعاد والتقسيم.

---

## ما الذي يتم عمل Bounding Box له؟

- Bounding Box مستقل لكل حرف سبئي ظاهر وقابل للتحديد.
- لا نضع Box واحدًا حول الكلمة.
- لا نضم حرفين في Box واحد.
- لا نترك حرفًا واضحًا داخل صورة تدريب بدون Annotation.
- لا نخمن Class لحرف غير قابل للتحديد بثقة؛ يوضع للمراجعة.
- الصورة الأصلية تبقى نظيفة؛ الـBounding Boxes تحفظ في ملفات Annotation منفصلة.

---

## صيغة Label

```text
class_id center_x center_y width height
```

القيم الهندسية normalized بين 0 و1.

---

## قواعد الجودة

1. الـBox يحيط بجميع أجزاء الحرف دون مساحة خلفية زائدة كبيرة.
2. لا يقطع جزءًا واضحًا من الحرف.
3. Class ID يراجع مقابل `config/classes.json`.
4. لا يوجد Box مكرر على نفس الحرف.
5. صور/قصاصات النقش نفسه لا توزع بين Train وTest لتجنب Data Leakage.
6. مصدر كل صورة وحق استخدامها يسجلان في `manifest.csv`.
7. لا يتم استخدام Roboflow augmentations أو صور مولدة ضمن Real Train.
8. الصور المكتوبة يدويًا يمكن أن تكون بيانات مساعدة/Pilot، لكن يجب أن تعكس المجموعة الأساسية المجال النهائي قدر الإمكان.

---

## قاعدة Class 29

`Class 29 = U+10A7D` لا يوضع لمجرد وجود خط عمودي.

لا يتم وسم:

- حافة ورقة.
- شق/خدش في الحجر.
- إطار أو ظل.
- أي خط عمودي غير مؤكد كرمز حقيقي.

أول Real Pilot أظهر أن Class 29 مسؤول عن جزء كبير من False Positives على الحواف والخطوط، لذلك هذه القاعدة إلزامية.

---

## Roboflow Class Names

أسماء الـClasses داخل Roboflow يجب أن تكون Project IDs نفسها بصيغة رقمين:

```text
00, 01, 02, ..., 29
```

لا نعتمد ترتيب Roboflow الداخلي كـProject ID؛ بعد التصدير نستخدم:

```bash
python scripts/remap_roboflow_labels.py \
  --dataset path/to/roboflow_export \
  --output path/to/fixed_export
```

الأداة تقرأ اسم الـClass وتعيد الـIDs إلى أرقام المشروع الثابتة 0..29 بدون تغيير الصور أو إحداثيات الـBoxes.

---

## التحقق قبل Fine-tuning

```bash
python scripts/validate_labels.py \
  --images data/real/images/train \
  --labels data/real/labels/train \
  --preview-dir outputs/real_label_previews
```

بعد ذلك تتم مراجعة previews بصريًا وتدقيق توزيع الـClasses والمصادر والتكرارات.

`finetune_real.py` يرفض البدء إذا كان Real Train أقل من 200 صورة أو كانت labels ناقصة/غير صالحة.

---

## سياسة التسليم

يجب إرسال أول **20 صورة** موسومة للمراجعة قبل إكمال بقية الـDataset. بعد قبول القاعدة، تُرسل دفعات أكبر بنفس الأسلوب. الهدف هو اكتشاف أي خطأ ثابت في الـBoxes أو Class naming مبكرًا قبل تكراره على مئات الصور.
