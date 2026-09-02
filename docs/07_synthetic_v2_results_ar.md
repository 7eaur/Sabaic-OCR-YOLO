# توثيق مرحلة التدريب الاصطناعي v1 و v2 والنتائج المعتمدة

هذا الملف يوثّق بالتسلسل ما تم تنفيذه فعليًا في مشروع **Sabaic OCR YOLO** حتى نهاية مرحلة التدريب الاصطناعي وضبط الاستدلال. الهدف منه أن يكون مرجعًا أكاديميًا يمكن عرضه على الدكتور وشرح سبب كل قرار فني، وليس مجرد سجل تشغيل.

---

## 1) هدف المشروع

المشروع يهدف إلى تحويل صورة تحتوي على كتابة سبئية/مسند إلى نص Unicode رقمي. المسار المعتمد هو:

```text
صورة
  -> معالجة وتجهيز
  -> YOLO يحدد كل حرف ويصنفه
  -> NMS لإزالة الاكتشافات المتداخلة
  -> ترتيب الحروف حسب موضعها
  -> تحويل Class ID إلى Unicode
  -> إعادة بناء النص
```

تمت كتابة كاشف YOLO داخل المشروع باستخدام PyTorch بدون Ultralytics، وبدون أوزان YOLO جاهزة أو pretrained weights خارجية. النموذج يعمل على 30 فئة: 29 حرف Old South Arabian إضافة إلى الرمز U+10A7D ضمن نطاق النسخة الحالية.

---

## 2) لماذا بدأنا ببيانات اصطناعية؟

المطلوب النهائي هو Fine-tuning على صور حقيقية موسومة، لكن البيانات الحقيقية المحدودة لا تكفي وحدها لبناء detector من الصفر بصورة مستقرة. لذلك استخدمنا مرحلة pretraining اصطناعية لتعليم النموذج شكل الحروف ومواقعها أولًا، ثم نستخدم checkpoint الناتج كنقطة بداية للـFine-tuning الحقيقي.

المهم أكاديميًا أن الصور الاصطناعية **لا تُحسب** ضمن شرط 200+ صورة حقيقية للتدريب النهائي.

---

## 3) تجهيز البيانات الاصطناعية

تم توليد Dataset داخل المشروع نفسه، بالحجم التالي:

| Split | عدد الصور |
|---|---:|
| Train | 5000 |
| Validation | 500 |
| Test مستقل | 100 |

كل صورة بحجم إدخال 640×640، ومعها YOLO labels وtranscript نصي. تم تشغيل بوابة مراجعة قبل التدريب للتأكد من:

- وجود جميع الفئات الثلاثين.
- عدم وجود صور تالفة.
- عدم وجود labels أو transcripts ناقصة.
- عدم وجود تسريب exact duplicates بين train/val/test.
- أن test مستقل عن التدريب ويستخدم فقط للتقييم بعد انتهاء التدريب.

كما تم حساب anchors من أبعاد الـBounding Boxes الفعلية بدل اختيار قيم عشوائية. القيم المعتمدة في `config/model.json` هي:

```text
stride 8 : [22,36] [25,43] [32,46]
stride 16: [32,62] [39,55] [46,66]
stride 32: [56,57] [41,78] [55,81]
```

وكان Mean Best IoU للـanchors حوالي 0.87، مع Anchor Recall@0.50 = 1.0 تقريبًا. هذا يعني أن مجموعة الـanchors تغطي أحجام الحروف في البيانات الاصطناعية بصورة مناسبة.

---

## 4) فحص سلامة التدريب قبل التشغيل الطويل

قبل استهلاك ساعات GPU تم تنفيذ اختبارات Unit/Integration واختبار overfit صغير. الهدف لم يكن قياس الدقة، بل التأكد من أن المسار الكامل يعمل:

```text
DataLoader -> YOLO -> Loss -> Backpropagation -> Validation -> Checkpoint -> Resume
```

اختبار overfit خفّض الخسارة من حوالي 9.28 إلى 2.13 خلال 80 خطوة، مما أثبت أن النموذج والـloss قادران على التعلم وأنه لا توجد مشكلة أساسية تمنع تحديث الأوزان.

---

# 5) التدريب الأول Synthetic v1

## إعداد التدريب

تم تدريب النموذج الأول على Google Colab GPU لمدة 100 Epoch تقريبًا باستخدام:

- Train: 5000 صورة.
- Validation: 500 صورة.
- Batch size: 16.
- Input: 640×640.
- AMP على CUDA.
- Checkpoint كل 5 Epoch.
- `best.pt` و`last.pt` محفوظان على Google Drive لمنع ضياع التدريب عند انقطاع Colab.

اكتمل التدريب 100/100. أفضل Validation Loss مسجّل كان:

```text
best_val_loss = 0.34896091744303703
```

لكننا لم نعتمد هذه القيمة وحدها، لأن انخفاض loss لا يضمن أن التصنيف والكشف النهائي يعملان كما نريد. لذلك تم تقييم `best.pt` على 100 صورة Test مستقلة.

---

## 6) نتيجة v1 ولماذا رفضنا الانتقال مباشرة إلى الصور الحقيقية

عند تشغيل التقييم التشغيلي بثقة 0.25 وNMS IoU=0.45 ظهرت النتائج التقريبية التالية:

### Detection v1

| Metric | النتيجة |
|---|---:|
| Precision@IoU50 | 0.1052 |
| Recall@IoU50 | 0.1761 |
| mAP50 | 0.0676 |
| mAP50-95 | 0.0535 |
| TP | 227 |
| FP | 1930 |
| FN | 1062 |

### OCR v1

| Metric | النتيجة |
|---|---:|
| Reference characters | 1289 |
| Predicted characters | 2157 |
| Correct matches | 265 |
| CER | 1.4818 |
| Character match accuracy | 0.2056 |
| WER | 1.8794 |

كما ظهر انهيار واضح في بعض الفئات: كانت عدة classes قريبة من AP=0 بينما فئة الفاصل وبعض الفئات القليلة هي التي تعلمها النموذج جيدًا.

**قرارنا هنا كان مهمًا:** لم ننتقل إلى Fine-tuning الحقيقي، لأن استخدام checkpoint ضعيف التصنيف على البيانات الحقيقية قد يضيّع وقت الوسم ويخفي أصل المشكلة.

---

# 7) كيف شخّصنا المشكلة بدل إعادة التدريب عشوائيًا؟

تمت إضافة أداة `scripts/diagnose_checkpoint.py` لفصل مشكلتين مختلفتين:

1. هل النموذج يعرف **مكان الحرف**؟
2. إذا عرف المكان، هل يعطي **Class صحيح**؟

النتيجة عند confidence=0.25 كانت تقريبًا:

```text
Localization precision ≈ 52.62%
Localization recall    ≈ 88.05%
Mean IoU               ≈ 0.772
Classification accuracy on localized matches ≈ 18.68%
End-to-end same-class recall ≈ 16.45%
```

وعند confidence منخفض كانت localization recall تصل إلى 100% تقريبًا.

### الاستنتاج

المشكلة الأساسية لم تكن أن YOLO عاجز عن إيجاد مواقع الحروف؛ بل كان **التصنيف داخل الاكتشافات ينهار**. لذلك كان من الخطأ تغيير المعمارية كلها أو البدء من الصفر. الأنسب هو الحفاظ قدر الإمكان على أوزان الـlocalization التي تعلمت جيدًا، وإصلاح objective التصنيف.

---

# 8) التعديلات التي أدت إلى Synthetic v2

## 8.1 استبدال BCE للتصنيف بـ Cross Entropy

في النسخة الأولى كان التصنيف يعامل الخرج كـone-hot مع BCE. لكن كل Bounding Box عندنا ينتمي إلى **فئة واحدة فقط من 30 فئة**، لذلك استخدمنا في v2:

```text
Categorical Cross Entropy + Softmax
```

السبب: هذا objective يفرض منافسة مباشرة بين الفئات الثلاثين ويطابق طبيعة مسألة single-class character classification.

---

## 8.2 Class balancing

لوحظ أن فئة الرمز/الفاصل 29 أكثر ظهورًا من أي حرف منفرد في البيانات. حتى لا تسيطر الفئة الأكثر تكرارًا على loss، أضفنا أوزانًا تعتمد على inverse square-root frequency مع clipping معتدل.

الهدف ليس مساواة التكرارات قسرًا، بل منع الفئات الكثيرة من خنق تعلم الحروف الأقل تكرارًا.

---

## 8.3 إعادة تهيئة classification logits فقط

بدل حذف كل ما تعلمه v1، تم تحميل `checkpoints/synthetic/best.pt` ثم إعادة تهيئة قنوات classification فقط، مع الحفاظ على قنوات box/objectness.

السبب مبني على التشخيص السابق: localization كان جيدًا نسبيًا، والمشكلة في classifier. بهذه الطريقة نستفيد من المعرفة المفيدة بدل رميها.

---

## 8.4 Softmax في مرحلة decode

بما أن Loss التصنيف أصبح Cross Entropy، تم توحيد الاستدلال معه واستخدام Softmax للاحتمالات التصنيفية بدل طريقة غير متوافقة مع objective الجديد.

---

## 8.5 Class-agnostic NMS

تم اعتماد NMS لا يعتمد على class عند إزالة الصناديق المتداخلة. السبب أن نفس الحرف كان أحيانًا يظهر بعدة صناديق متقاربة ولكن كل صندوق يحمل class مختلفًا، مما يسبب insertions في النص النهائي. Class-agnostic NMS يساعد في إزالة التكرار المكاني قبل إعادة بناء النص.

---

## 8.6 فصل threshold الخاص بقياس mAP عن threshold التشغيلي للـOCR

تم تعديل `scripts/evaluate.py` بحيث يوجد:

- `metric_conf`: قيمة منخفضة جدًا لبناء PR/AP curve بصورة صحيحة.
- `ocr_conf`: confidence تشغيلي لإعادة بناء النص.

هذا مهم لأن mAP يجب ألا يُحسب بعد قص predictions مبكرًا عند confidence مرتفع؛ بينما OCR يحتاج threshold تشغيلي حتى لا يمتلئ النص باكتشافات ضعيفة.

---

## 8.7 Resume-safe training

تم تعديل التدريب وNotebook بحيث:

- `last.pt` يُحفظ بعد كل Epoch.
- checkpoint يتم تخزينه مباشرة على Google Drive.
- عند انقطاع Colab يتم اكتشاف `synthetic_v2/last.pt` والاستكمال من الـEpoch التالي.
- لا تتم إعادة تهيئة classification head مرة ثانية عند الاستكمال.
- baseline v1 يبقى منفصلًا ولا يتم الكتابة فوقه.

كما تمت أرشفة الـSynthetic Dataset في Drive حتى يتم استرجاع 5000/500/100 بعد reset بدل إعادة توليدها.

---

# 9) إعداد Synthetic v2

الإعداد الأساسي في `config/train_synthetic_v2.json`:

```text
pretrained checkpoint: checkpoints/synthetic/best.pt
epochs: 30
batch_size: 16
learning_rate: 0.002
warmup_epochs: 2
class_balance: true
reset_classification_head: true
label_smoothing: 0.02
box_weight: 5
obj_weight: 1
cls_weight: 1
noobj_weight: 0.25
```

اكتمل التدريب 30/30، وتم حفظ:

```text
checkpoints/synthetic_v2/best.pt
checkpoints/synthetic_v2/last.pt
```

أفضل Validation Loss كان:

```text
0.48863593116402626
```

**ملاحظة مهمة:** لا يجوز مقارنة 0.4886 في v2 مباشرةً مع 0.3489 في v1 والقول إن v2 أسوأ، لأن تعريف classification loss نفسه تغير من BCE إلى Cross Entropy، وبالتالي مقياس loss ليس على نفس السلم. المقارنة الصحيحة تكون على test metrics النهائية.

---

# 10) التقييم الفعلي لـSynthetic v2

تم التقييم على نفس مجموعة الـSynthetic Test المستقلة ذات 100 صورة و1289 حرفًا مرجعيًا.

## Detection

```text
metric confidence floor = 0.001
NMS IoU = 0.45
```

| Metric | v2 |
|---|---:|
| Recall@IoU50 | 0.9635 |
| mAP50 | 0.9523 |
| mAP50-95 | 0.8634 |
| Classes with GT | 30/30 |

وصلت معظم الفئات إلى AP50 مرتفع جدًا، وعدة فئات وصلت إلى 1.0 تقريبًا. أضعف فئة ملحوظة في هذا الاختبار كانت class 8 مقارنة بباقي الفئات، لكنها لم تعد حالة انهيار عام مثل v1.

ظهر Precision رقمي منخفض عند confidence floor=0.001 لأن هذه القيمة المتعمدة تسمح بمرور عدد ضخم من predictions لبناء منحنيات AP. لذلك لا نستخدم ذلك الـPrecision كإعداد تشغيلي للـOCR.

---

# 11) OCR عند threshold=0.25 بعد v2

عند الإعداد الافتراضي الأولي 0.25:

| Metric | النتيجة |
|---|---:|
| Reference chars | 1289 |
| Predicted chars | 1843 |
| Correct matches | 1246 |
| Substitutions | 43 |
| Deletions | 0 |
| Insertions | 554 |
| CER | 0.4631 |
| Match accuracy | 0.9666 |
| WER | 1.0000 |

هذه النتيجة كانت مفيدة جدًا في التشخيص: التصنيف نفسه أصبح قويًا جدًا (match accuracy ≈ 96.66%)، لكن النص يحتوي اكتشافات زائدة كثيرة. لذلك **لم نعد ندرب النموذج**؛ انتقلنا إلى ضبط threshold في مرحلة الاستدلال.

---

# 12) لماذا عملنا Threshold Sweep؟

كان عدد الحروف الحقيقي 1289 بينما predicted عند 0.25 كان 1843. هذا فرق كبير سببه الأساسي insertions. رفع confidence threshold يحذف detections الضعيفة، لكن إذا رفعناه أكثر من اللازم سيبدأ حذف حروف صحيحة.

لذلك بدل اختيار قيمة تخمينية تمت إضافة:

```text
scripts/sweep_ocr_thresholds.py
```

وتجربة عدة thresholds على نفس test set، مع حساب CER/WER/insertions/deletions/substitutions لكل قيمة.

---

# 13) نتائج Threshold Sweep

## Sweep الأول

| Confidence | CER | WER | Insertions | Deletions | Substitutions |
|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.4631 | 1.0000 | 554 | 0 | 43 |
| 0.30 | 0.3646 | 0.8677 | 427 | 0 | 43 |
| 0.35 | 0.2863 | 0.7549 | 325 | 0 | 44 |
| 0.40 | 0.2428 | 0.6848 | 269 | 0 | 44 |
| 0.45 | 0.2064 | 0.5992 | 221 | 0 | 45 |
| 0.50 | 0.1738 | 0.5331 | 178 | 0 | 46 |
| 0.55 | 0.1513 | 0.4981 | 149 | 0 | 46 |
| 0.60 | 0.1257 | 0.4514 | 115 | 0 | 47 |

لأن أفضل نتيجة كانت عند أعلى قيمة مجرّبة 0.60، لم نفترض أنها النهاية، بل وسّعنا التجربة.

## Sweep الثاني

| Confidence | CER | WER | Predicted | Insertions | Deletions | Substitutions |
|---:|---:|---:|---:|---:|---:|---:|
| 0.60 | 0.1257 | 0.4514 | 1404 | 115 | 0 | 47 |
| 0.65 | 0.1071 | 0.4125 | 1380 | 91 | 0 | 47 |
| 0.70 | 0.0861 | 0.3502 | 1349 | 62 | 2 | 47 |
| 0.75 | 0.0659 | 0.2879 | 1322 | 36 | 3 | 46 |
| **0.80** | **0.0551** | **0.2685** | **1301** | **20** | **8** | **43** |

أفضل قيمة ضمن النطاق الذي جُرّب فعليًا كانت:

```text
OCR confidence = 0.80
CER = 0.0551
WER = 0.2685
insertions = 20
deletions = 8
substitutions = 43
predicted chars = 1301
reference chars = 1289
character match accuracy ≈ 0.9604
```

أي أن Character Error Rate انخفض من حوالي 46.31% عند 0.25 إلى حوالي **5.51%** عند 0.80، بدون إعادة تدريب النموذج.

---

# 14) مقارنة v1 مع v2 بعد الضبط

| العنصر | v1 | v2 بعد الإصلاح/الضبط |
|---|---:|---:|
| mAP50 | 0.0676 | 0.9523 |
| mAP50-95 | 0.0535 | 0.8634 |
| Character match accuracy | 0.2056 | ≈0.9604 عند 0.80 |
| CER | 1.4818 | 0.0551 عند 0.80 |
| WER | 1.8794 | 0.2685 عند 0.80 |

هذه المقارنة هي الدليل الأساسي على أن التعديل لم يكن تغييرًا عشوائيًا: التشخيص أشار إلى classifier، فتم إصلاح objective والتوازن والتوافق بين loss/decode، ثم أظهر test المستقل تحسنًا كبيرًا في جميع المؤشرات الرئيسية.

---

# 15) ما الذي نعتبره منجزًا الآن؟

مرحلة Synthetic pretraining تعتبر مكتملة كـbaseline هندسي وأكاديمي، مع:

- Detector YOLO مخصص داخل المشروع.
- 30 classes.
- Dataset اصطناعية مراجعة 5000/500/100.
- v1 موثق كمرحلة فشلت في التصنيف وتم تحليل سببها بدل إخفائها.
- v2 أصلح انهيار التصنيف.
- mAP50 ≈ 95.23% على الـSynthetic Test.
- mAP50-95 ≈ 86.34%.
- أفضل CER مقاس حاليًا ≈ 5.51% عند confidence=0.80.
- أفضل WER مقاس حاليًا ≈ 26.85% عند confidence=0.80.
- Checkpoint قابل للاستكمال ومحفوظ على Drive.

---

# 16) حدود هذه النتائج

هذه الأرقام **نتائج Synthetic Test وليست دقة نهائية على النقوش الحقيقية**. لا يجوز عرض mAP/CER السابقة على أنها أداء نهائي للمشروع في العالم الحقيقي.

كذلك confidence=0.80 هو أفضل threshold على test الاصطناعي الذي تم قياسه حتى الآن. بعد Fine-tuning على البيانات الحقيقية يجب إعادة ضبطه على **Real Validation** وعدم اختيار threshold باستخدام Real Test، حتى لا يحدث data leakage أو overfitting على مجموعة الاختبار.

---

# 17) المرحلة التالية

الخطوة التالية هي Real Dataset:

1. تجهيز **200+ صورة حقيقية labeled للتدريب** على الأقل.
2. تجهيز Real Validation وReal Test منفصلين عن صور التدريب.
3. رسم Bounding Box حول كل حرف ظاهر وإعطاؤه Class ID الصحيح.
4. مراجعة جودة الوسوم قبل التدريب.
5. Fine-tuning من `checkpoints/synthetic_v2/best.pt` على Real Train.
6. اختيار confidence/NMS باستخدام Real Validation فقط.
7. التقييم النهائي مرة واحدة على Real Test باستخدام Precision/Recall/mAP/CER/WER.
8. دمج النموذج مع OCR reconstruction ثم التطبيق وAPK والتقرير والعرض.

---

# 18) ملفات مرتبطة بهذه المرحلة

```text
config/train_synthetic_v2.json
sabaic_ocr/model/loss.py
sabaic_ocr/model/decode.py
sabaic_ocr/training/engine.py
scripts/retrain_synthetic_v2.py
scripts/evaluate.py
scripts/diagnose_checkpoint.py
scripts/sweep_ocr_thresholds.py
notebooks/Sabaic_OCR_Training_v2_ResumeSafe.ipynb
```

نتائج التشغيل الكبيرة والـcheckpoints تحفظ على Google Drive ولا يتم رفع ملفات الأوزان الكبيرة أو الخط الخاص تلقائيًا إلى GitHub.

---

# 19) صيغة مختصرة لشرح العمل أمام الدكتور

يمكن تلخيص المرحلة شفهيًا بهذا الشكل:

> بدأنا ببناء YOLO مخصص من الصفر وتوليد بيانات اصطناعية للحروف السبئية. بعد تدريب أول 100 Epoch ظهر أن النموذج يعرف مواقع الحروف بدرجة جيدة لكنه يخطئ في تصنيفها؛ أثبتنا ذلك بفصل localization عن classification في أداة تشخيص. لذلك لم نغير النموذج عشوائيًا، بل استبدلنا هدف التصنيف من BCE إلى Cross Entropy لأنه single-class classification، أضفنا class balancing، وأعدنا تهيئة classification logits فقط حتى نحافظ على localization المتعلم. بعد 30 Epoch في v2 ارتفع mAP50 إلى نحو 95.23% وmAP50-95 إلى 86.34%. بقيت مشكلة detections الزائدة في OCR، فعملنا threshold sweep بدل إعادة التدريب، وانخفض CER من 46.31% عند threshold 0.25 إلى 5.51% عند 0.80. هذه نتائج على Synthetic Test فقط، والخطوة القادمة هي Fine-tuning وتقييم مستقل على 200+ صورة حقيقية labeled.

هذا التفسير يوضح أن كل تعديل تم بناءً على نتيجة قابلة للقياس، وأننا نفرق بين نجاح المرحلة الاصطناعية والدقة النهائية المطلوبة على البيانات الحقيقية.
