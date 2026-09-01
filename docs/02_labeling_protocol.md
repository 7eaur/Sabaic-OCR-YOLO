# بروتوكول الـLabelling للصور الحقيقية

## شرط Fine-tuning

`data/real/images/train` يجب أن يحتوي **200 صورة حقيقية على الأقل**. الصور الاصطناعية لا تدخل في هذا العدد. نوصي بصور Real Validation وReal Test إضافية مستقلة.

## ما الذي يتم عمل Bounding Box له؟

- Bounding Box مستقل لكل حرف سبئي ظاهر وقابل للتحديد.
- لا نضع Box واحدًا حول الكلمة.
- لا يشترط عدد كبير من الحروف في الصورة الواحدة.
- لا نخمن Class لحرف غير قابل للتحديد بثقة؛ يوضع للمراجعة.

## صيغة Label

```text
class_id center_x center_y width height
```

القيم الهندسية normalized بين 0 و1.

## قواعد الجودة

1. الـBox يحيط بالحرف دون مساحة زائدة كبيرة.
2. لا يقطع جزءًا واضحًا من الحرف.
3. Class ID يراجع مقابل `config/classes.json`.
4. صور/قصاصات النقش نفسه لا توزع بين Train وTest لتجنب Data Leakage.
5. مصدر كل صورة وحق استخدامها يسجلان في manifest.

## التحقق

```bash
python scripts/validate_labels.py \
  --images data/real/images/train \
  --labels data/real/labels/train \
  --preview-dir outputs/real_label_previews
```

`finetune_real.py` يرفض البدء إذا كان Real Train أقل من 200 صورة أو كانت labels ناقصة/غير صالحة.
