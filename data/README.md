# Dataset Layout

```text
data/
├── synthetic/
│   ├── images/train
│   ├── images/val
│   ├── labels/train
│   ├── labels/val
│   └── transcripts/...
└── real/
    ├── images/train
    ├── images/val
    ├── images/test
    ├── labels/train
    ├── labels/val
    ├── labels/test
    ├── transcripts/train
    ├── transcripts/val
    └── transcripts/test
```

`real/images/train` يجب أن يحتوي على 200 صورة حقيقية labeled على الأقل قبل Fine-tuning. يجب تسجيل مصدر كل صورة وترخيص/حق استخدامها.
