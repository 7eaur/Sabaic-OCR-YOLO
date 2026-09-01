# الاختبار والتقييم

## Unit tests

```bash
python -m unittest discover -s tests -v
```

## Detection

على Real Test فقط:

- Precision @ IoU 0.50
- Recall @ IoU 0.50
- AP50 لكل Class
- mAP@0.50
- mAP@0.50:0.95

التنفيذ داخلي في `sabaic_ocr/metrics/detection.py`.

## OCR — Character level

- Correct characters
- Wrong characters
- Substitutions
- Deletions
- Insertions
- CER
- Accuracy derived from CER

## OCR — Word level

- Correct words
- Wrong words
- Substitutions
- Deletions
- Insertions
- WER
- Accuracy derived from WER

## لا توجد نتائج مختلقة

أي رقم في التقرير النهائي يجب أن يأتي من تشغيل فعلي مثل:

```bash
python scripts/evaluate.py --checkpoint checkpoints/real_finetune/best.pt
```

ويحفظ JSON الناتج كدليل على التجربة.
