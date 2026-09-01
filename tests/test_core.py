import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from sabaic_ocr.data.dataset import YoloCharacterDataset
from sabaic_ocr.metrics.ocr import evaluate_pair
from sabaic_ocr.model.box_ops import xywh_to_xyxy, xyxy_to_xywh
from sabaic_ocr.model.decode import postprocess_batch
from sabaic_ocr.model.loss import YoloLoss
from sabaic_ocr.model.yolo import SabaicYOLO
from sabaic_ocr.ocr.postprocess import OCRDetection, detections_to_text


class CoreTests(unittest.TestCase):
    def test_box_roundtrip(self):
        x = torch.tensor([[0.5, 0.5, 0.2, 0.4]], dtype=torch.float32)
        y = xyxy_to_xywh(xywh_to_xyxy(x))
        self.assertTrue(torch.allclose(x, y, atol=1e-6))

    def test_model_shapes_and_loss(self):
        model = SabaicYOLO(num_classes=30, width_mult=0.25, depth_mult=0.25)
        x = torch.randn(1, 3, 128, 128)
        p = model(x)
        self.assertEqual(len(p), 3)
        self.assertEqual(p[0].shape[-2:], (16, 16))
        self.assertEqual(p[1].shape[-2:], (8, 8))
        self.assertEqual(p[2].shape[-2:], (4, 4))

        anchors = [
            [[4, 8], [6, 12], [8, 16]],
            [[10, 20], [14, 28], [18, 36]],
            [[24, 48], [32, 64], [48, 80]],
        ]
        criterion = YoloLoss(anchors, 30, image_size=128)
        targets = [torch.tensor([[3, 0.5, 0.5, 0.08, 0.16]], dtype=torch.float32)]
        loss = criterion(p, targets)
        self.assertTrue(torch.isfinite(loss.total))
        self.assertEqual(loss.positives, 1)

        decoded = postprocess_batch(
            p, anchors, 30, image_size=128, conf_threshold=0.9999
        )
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0].shape[1], 6)

    def test_ocr_order_rtl(self):
        mapping = {0: "A", 1: "B", 2: "C"}
        dets = [
            OCRDetection(0, .9, 0.7, 0.1, 0.8, 0.2),
            OCRDetection(1, .9, 0.5, 0.1, 0.6, 0.2),
            OCRDetection(2, .9, 0.3, 0.1, 0.4, 0.2),
        ]
        self.assertEqual(detections_to_text(dets, mapping, "rtl"), "ABC")

    def test_ocr_metrics(self):
        r = evaluate_pair("ABC", "ADC")
        self.assertEqual(r["character"]["correct"], 2)
        self.assertEqual(r["character"]["wrong"], 1)
        self.assertAlmostEqual(r["character"]["cer"], 1 / 3)
        self.assertAlmostEqual(r["character"]["match_accuracy"], 2 / 3)

    def test_dataset_load(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            (td / "images").mkdir()
            (td / "labels").mkdir()
            Image.new("RGB", (100, 50), "white").save(td / "images" / "a.jpg")
            (td / "labels" / "a.txt").write_text(
                "0 0.5 0.5 0.2 0.4\n", encoding="utf-8"
            )
            ds = YoloCharacterDataset(td / "images", td / "labels", 30, 128)
            image, targets = ds[0]
            self.assertEqual(tuple(image.shape), (3, 128, 128))
            self.assertEqual(tuple(targets.shape), (1, 5))


if __name__ == "__main__":
    unittest.main()


def test_dense_same_cell_uses_distinct_anchor_slots():
    """Two nearby GTs must not create contradictory targets on one raw prediction."""
    model = SabaicYOLO(num_classes=30, width_mult=0.25, depth_mult=0.25)
    x = torch.randn(1, 3, 128, 128)
    p = model(x)
    anchors = [
        [[4, 8], [6, 12], [8, 16]],
        [[10, 20], [14, 28], [18, 36]],
        [[24, 48], [32, 64], [48, 80]],
    ]
    criterion = YoloLoss(anchors, 30, image_size=128)
    targets = [torch.tensor([
        [3, 0.50, 0.50, 0.08, 0.16],
        [4, 0.50, 0.50, 0.08, 0.16],
    ], dtype=torch.float32)]
    loss = criterion(p, targets)
    assert torch.isfinite(loss.total)
    assert loss.positives == 2


def test_word_metric_does_not_treat_linebreak_as_word_boundary():
    from sabaic_ocr.metrics.ocr import words_from_sabaic
    assert words_from_sabaic("AB\nCD𐩽EF") == ["ABCD", "EF"]


def test_label_rejects_box_outside_image():
    from sabaic_ocr.data.dataset import read_yolo_labels
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "bad.txt"
        p.write_text("0 0.05 0.5 0.2 0.2\n", encoding="utf-8")
        with __import__('pytest').raises(ValueError, match="outside image bounds"):
            read_yolo_labels(p, 30)
