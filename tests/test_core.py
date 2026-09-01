import tempfile,unittest
from pathlib import Path
import torch
from PIL import Image
from sabaic_ocr.data.dataset import YoloCharacterDataset
from sabaic_ocr.metrics.ocr import evaluate_pair
from sabaic_ocr.model.box_ops import xywh_to_xyxy,xyxy_to_xywh
from sabaic_ocr.model.decode import postprocess_batch
from sabaic_ocr.model.loss import YoloLoss
from sabaic_ocr.model.yolo import SabaicYOLO
from sabaic_ocr.ocr.postprocess import OCRDetection,detections_to_text

class CoreTests(unittest.TestCase):
    def test_box_roundtrip(self):
        x=torch.tensor([[0.5,0.5,0.2,0.4]]); self.assertTrue(torch.allclose(x,xyxy_to_xywh(xywh_to_xyxy(x)),atol=1e-6))
    def test_model_shapes_and_loss(self):
        model=SabaicYOLO(30,0.25,0.25); p=model(torch.randn(1,3,128,128)); self.assertEqual([x.shape[-2:] for x in p],[(16,16),(8,8),(4,4)])
        anchors=[[[4,8],[6,12],[8,16]],[[10,20],[14,28],[18,36]],[[24,48],[32,64],[48,80]]]; loss=YoloLoss(anchors,30,128)(p,[torch.tensor([[3,0.5,0.5,0.08,0.16]])]); self.assertTrue(torch.isfinite(loss.total)); self.assertEqual(loss.positives,1); self.assertEqual(postprocess_batch(p,anchors,30,128,0.9999)[0].shape[1],6)
    def test_ocr_order_rtl(self):
        dets=[OCRDetection(0,.9,.7,.1,.8,.2),OCRDetection(1,.9,.5,.1,.6,.2),OCRDetection(2,.9,.3,.1,.4,.2)]; self.assertEqual(detections_to_text(dets,{0:"A",1:"B",2:"C"},"rtl"),"ABC")
    def test_ocr_metrics(self):
        r=evaluate_pair("ABC","ADC"); self.assertEqual(r["character"]["correct"],2); self.assertEqual(r["character"]["wrong"],1); self.assertAlmostEqual(r["character"]["cer"],1/3)
    def test_dataset_load(self):
        with tempfile.TemporaryDirectory() as td:
            td=Path(td); (td/"images").mkdir(); (td/"labels").mkdir(); Image.new("RGB",(100,50),"white").save(td/"images"/"a.jpg"); (td/"labels"/"a.txt").write_text("0 0.5 0.5 0.2 0.4\n",encoding="utf-8"); image,targets=YoloCharacterDataset(td/"images",td/"labels",30,128)[0]; self.assertEqual(tuple(image.shape),(3,128,128)); self.assertEqual(tuple(targets.shape),(1,5))

if __name__=="__main__": unittest.main()
