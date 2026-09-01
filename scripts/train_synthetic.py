#!/usr/bin/env python3
import argparse,json
from sabaic_ocr.config import load_json
from sabaic_ocr.training.engine import train_detector


def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="config/train_synthetic.json"); args=p.parse_args(); train_cfg=load_json(args.config); model_cfg=load_json(train_cfg["model_config"]); result=train_detector(train_cfg,model_cfg); print(json.dumps({k:v for k,v in result.items() if k!="history"},indent=2))

if __name__=="__main__": main()
