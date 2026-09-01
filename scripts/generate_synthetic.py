#!/usr/bin/env python3
from __future__ import annotations

import argparse,json
from pathlib import Path
from sabaic_ocr.config import load_classes,load_json
from sabaic_ocr.data.synthetic import SyntheticSabaicGenerator,load_corpus,save_sample


def generate_split(generator,root,split,count,start_index=0):
    for i in range(count):
        sample=generator.make_sample(); name=f"sabaic_{split}_{start_index+i:06d}"
        save_sample(sample,root/"images"/split/f"{name}.jpg",root/"labels"/split/f"{name}.txt",root/"transcripts"/split/f"{name}.txt")
        if (i+1)%250==0 or i+1==count: print(f"{split}: {i+1}/{count}")


def main():
    p=argparse.ArgumentParser(description="Generate custom synthetic Sabaic YOLO dataset.")
    p.add_argument("--output",default="data/synthetic"); p.add_argument("--font",default="assets/fonts/NotoSansOldSouthArabian-Regular.ttf"); p.add_argument("--classes",default="config/classes.json"); p.add_argument("--model-config",default="config/model.json"); p.add_argument("--corpus",default=""); p.add_argument("--train",type=int,default=10000); p.add_argument("--val",type=int,default=1500); p.add_argument("--test",type=int,default=0); p.add_argument("--seed",type=int,default=42); args=p.parse_args()
    classes=load_classes(args.classes); model_cfg=load_json(args.model_config); corpus=load_corpus(args.corpus if args.corpus else None); root=Path(args.output)
    gen=SyntheticSabaicGenerator(classes["classes"],args.font,int(model_cfg["image_size"]),args.seed,corpus)
    generate_split(gen,root,"train",args.train,0); generate_split(gen,root,"val",args.val,args.train)
    if args.test>0: generate_split(gen,root,"test",args.test,args.train+args.val)
    manifest={"generator":"custom_internal_generator","train_images":args.train,"val_images":args.val,"test_images":args.test,"image_size":model_cfg["image_size"],"seed":args.seed,"font_path":args.font,"corpus_path":args.corpus or None,"note":"Random sequences are detector-training data and are not claimed to be historically valid Sabaic unless an externally verified corpus is supplied."}
    root.mkdir(parents=True,exist_ok=True); (root/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(manifest,ensure_ascii=False,indent=2))

if __name__=="__main__": main()
