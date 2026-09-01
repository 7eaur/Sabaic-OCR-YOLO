# Detector Training Phase — Status

The synthetic-data stage has passed its gate and the project has moved into the detector-training phase.

## Full-loop preflight executed

Before spending GPU hours, the complete training entry point was exercised on a deliberately small smoke dataset and reduced model/input size. This verified DataLoader -> custom YOLO -> custom loss -> optimizer -> validation -> checkpoint writing -> resume.

First smoke epoch:

```text
epoch 1/1 train=9.9365 val=9.4785
```

The run wrote `best.pt`, `last.pt` and an epoch checkpoint. A second invocation then loaded `last.pt` and resumed at the next epoch rather than restarting:

```text
epoch 2/2 train=9.5263 val=10.5408
```

This is a **pipeline/resume test only**. These loss values are not project accuracy and must not be presented as final results.

## Repository CI

GitHub Actions now compiles `sabaic_ocr` and `scripts` and runs the unit test suite on the exact committed repository. The first CI run completed successfully after the stage-review changes.

## Full training target

The actual synthetic pretraining configuration is:

- 5,000 reviewed Train images.
- 500 reviewed Validation images.
- 640 x 640 model input.
- 30 classes.
- 100 configured epochs.
- batch size 16.
- AMP on CUDA.
- checkpoints persisted as `checkpoints/synthetic/best.pt` and `last.pt`.

The Colab notebook mounts Google Drive for persistent checkpoints and asserts that a GPU is available before the full training cell starts.

## Not yet claimed

A full GPU run has **not** been executed in the current CPU-only execution environment. Therefore there is no legitimate final synthetic `best.pt`, mAP, character accuracy, CER, word accuracy or WER to report yet. Those values must come from the actual GPU training/evaluation run.
