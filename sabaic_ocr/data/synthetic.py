from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageFont


@dataclass
class RenderedSample:
    image: Image.Image
    labels: List[Tuple[int,float,float,float,float]]
    transcription: str


def _procedural_stone(width, height, rng):
    base = int(rng.integers(155,225))
    low_h, low_w = max(4,height//24), max(4,width//24)
    low = rng.normal(0,24,size=(low_h,low_w)).astype(np.float32)
    low_img = Image.fromarray(np.clip(low+128,0,255).astype(np.uint8), mode="L").resize((width,height), Image.Resampling.BICUBIC)
    low_arr = np.asarray(low_img).astype(np.float32)-128.0
    fine = rng.normal(0,7,size=(height,width)).astype(np.float32)
    arr = np.clip(base + low_arr*0.55 + fine, 70,245).astype(np.uint8)
    return Image.fromarray(np.repeat(arr[...,None],3,axis=2), mode="RGB")


class SyntheticSabaicGenerator:
    """Custom synthetic generator that returns exact character boxes without TRDG."""
    def __init__(self, classes: Sequence[dict], font_path, image_size: int=640, seed: int=42, corpus_lines: Sequence[str]|None=None):
        self.classes = list(classes)
        self.font_path = Path(font_path)
        self.image_size = int(image_size)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        if not self.font_path.exists():
            raise FileNotFoundError(f"Font not found: {self.font_path}. Place NotoSansOldSouthArabian-Regular.ttf in assets/fonts/.")
        self.char_to_class: Dict[str,int] = {c["char"]:int(c["id"]) for c in self.classes}
        self.letter_chars = [c["char"] for c in self.classes if c.get("type")=="letter"]
        self.separator_char = next((c["char"] for c in self.classes if c.get("type")=="number_or_separator"), "𐩽")
        self.allowed_chars = set(self.char_to_class)
        self.corpus_lines = [self._filter_text(x) for x in (corpus_lines or []) if self._filter_text(x)]

    def _filter_text(self,text):
        return "".join(ch for ch in text.strip() if ch in self.allowed_chars or ch.isspace())

    def _random_line(self):
        words=[]
        for _ in range(self.rng.randint(1,4)):
            words.append("".join(self.rng.choice(self.letter_chars) for _ in range(self.rng.randint(2,6))))
        return self.separator_char.join(words)

    def _choose_text(self):
        if self.corpus_lines and self.rng.random()<0.75:
            text=self.separator_char.join(self.rng.choice(self.corpus_lines).split())
            text="".join(ch for ch in text if ch in self.allowed_chars)
            if text:
                return text
        return self._random_line()

    def _background(self):
        if self.rng.random()<0.75:
            return _procedural_stone(self.image_size,self.image_size,self.np_rng)
        v=int(self.np_rng.integers(185,246))
        return Image.new("RGB",(self.image_size,self.image_size),(v,v,v))

    def _render_line(self, draw, text, font, y_top, right_margin, left_margin, spacing, fill):
        cursor_x=self.image_size-right_margin
        boxes=[]; kept=[]
        for ch in text:
            if ch not in self.char_to_class:
                continue
            bbox0=draw.textbbox((0,0),ch,font=font)
            gw=max(1,bbox0[2]-bbox0[0]); gh=max(1,bbox0[3]-bbox0[1])
            left=cursor_x-gw; top=y_top; right=cursor_x; bottom=y_top+gh
            if left<left_margin:
                break
            draw.text((left-bbox0[0], top-bbox0[1]), ch, font=font, fill=fill)
            boxes.append((self.char_to_class[ch],left,top,right,bottom)); kept.append(ch)
            cursor_x=left-spacing
        return boxes,"".join(kept)

    def make_sample(self):
        image=self._background(); draw=ImageDraw.Draw(image)
        font_size=self.rng.randint(44,92); font=ImageFont.truetype(str(self.font_path),font_size)
        line_count=self.rng.choices([1,2,3],weights=[0.55,0.32,0.13])[0]
        top_margin=self.rng.randint(35,80); right_margin=self.rng.randint(30,70); left_margin=self.rng.randint(30,70)
        spacing=self.rng.randint(4,18); line_gap=self.rng.randint(28,65); ink=self.rng.randint(20,95); fill=(ink,ink,ink)
        all_boxes=[]; rendered=[]; y=top_margin
        for _ in range(line_count):
            boxes, kept=self._render_line(draw,self._choose_text(),font,y,right_margin,left_margin,spacing,fill)
            if boxes:
                all_boxes.extend(boxes); rendered.append(kept)
            y += font_size+line_gap
            if y+font_size>=self.image_size-20:
                break
        if self.rng.random()<0.55:
            image=ImageEnhance.Contrast(image).enhance(self.rng.uniform(0.72,1.35))
        if self.rng.random()<0.45:
            image=ImageEnhance.Brightness(image).enhance(self.rng.uniform(0.82,1.15))
        if self.rng.random()<0.42:
            image=image.filter(ImageFilter.GaussianBlur(self.rng.uniform(0.15,1.25)))
        if self.rng.random()<0.65:
            arr=np.asarray(image).astype(np.float32); arr += self.np_rng.normal(0.0,self.rng.uniform(1.0,8.5),size=arr.shape)
            image=Image.fromarray(np.clip(arr,0,255).astype(np.uint8),mode="RGB")
        labels=[]
        for cls_id,x1,y1,x2,y2 in all_boxes:
            x1=max(0,min(self.image_size-1,x1)); y1=max(0,min(self.image_size-1,y1)); x2=max(x1+1,min(self.image_size,x2)); y2=max(y1+1,min(self.image_size,y2))
            labels.append((cls_id,((x1+x2)/2)/self.image_size,((y1+y2)/2)/self.image_size,(x2-x1)/self.image_size,(y2-y1)/self.image_size))
        return RenderedSample(image=image,labels=labels,transcription="\n".join(rendered))


def load_corpus(path):
    if not path:
        return []
    path=Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return [x.strip() for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def save_sample(sample, image_path, label_path, transcript_path):
    image_path,label_path,transcript_path=Path(image_path),Path(label_path),Path(transcript_path)
    image_path.parent.mkdir(parents=True,exist_ok=True); label_path.parent.mkdir(parents=True,exist_ok=True); transcript_path.parent.mkdir(parents=True,exist_ok=True)
    sample.image.save(image_path,quality=92)
    lines=[f"{c} {x:.8f} {y:.8f} {w:.8f} {h:.8f}" for c,x,y,w,h in sample.labels]
    label_path.write_text("\n".join(lines)+( "\n" if lines else ""),encoding="utf-8")
    transcript_path.write_text(sample.transcription,encoding="utf-8")
