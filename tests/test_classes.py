import json,unittest
from pathlib import Path

class ClassMappingTests(unittest.TestCase):
    def test_mapping(self):
        cfg=json.loads(Path("config/classes.json").read_text(encoding="utf-8")); self.assertEqual(cfg["num_classes"],30); self.assertEqual([x["id"] for x in cfg["classes"]],list(range(30))); self.assertEqual(cfg["classes"][0]["unicode"],"U+10A60"); self.assertEqual(cfg["classes"][28]["unicode"],"U+10A7C"); self.assertEqual(cfg["classes"][29]["unicode"],"U+10A7D"); self.assertEqual(len({x["char"] for x in cfg["classes"]}),30)

if __name__=="__main__": unittest.main()
