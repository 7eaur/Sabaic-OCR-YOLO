from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageEnhance, ImageFilter, ImageFont


@dataclass
class RenderedSample:
    image: Image.Image
    labels: List[Tuple[int, float, float, float, float]]
    transcription: str


def _procedural_stone(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    """Fast multi-scale procedural stone texture.

    Noise is created at low/mid resolution and upsampled with Pillow. This is
    much faster than sampling several full 640x640 floating-point arrays for
    every training image while preserving useful low-frequency stone texture.
    """
    base = int(rng.integers(145, 225))
    low_h = max(8, height // 24)
    low_w = max(8, width // 24)
    mid_h = max(24, height // 5)
    mid_w = max(24, width // 5)

    low = np.clip(rng.normal(base, 26, size=(low_h, low_w)), 55, 245).astype(np.uint8)
    mid = np.clip(rng.normal(base, 12, size=(mid_h, mid_w)), 55, 245).astype(np.uint8)
    low_img = Image.fromarray(low, mode="L").resize((width, height), Image.Resampling.BICUBIC)
    mid_img = Image.fromarray(mid, mode="L").resize((width, height), Image.Resampling.BILINEAR)
    gray = Image.blend(low_img, mid_img, 0.35)
    gray = ImageEnhance.Contrast(gray).enhance(float(rng.uniform(0.8, 1.3)))
    return gray.convert("RGB")


def _plain_background(width: int, height: int, rng: np.random.Generator) -> Image.Image:
    value = int(rng.integers(175, 246))
    return Image.new("RGB", (width, height), (value, value, value))


class SyntheticSabaicGenerator:
    """
    Internal synthetic generator for character-level YOLO training.

    The generator renders each Old South Arabian glyph separately from right to
    left, so its bounding box is known exactly at creation time. It deliberately
    includes short and long samples because the real fine-tuning images are not
    required to contain many characters.
    """

    def __init__(
        self,
        classes: Sequence[dict],
        font_path: str | Path,
        image_size: int = 640,
        seed: int = 42,
        corpus_lines: Sequence[str] | None = None,
    ):
        self.classes = list(classes)
        self.font_path = Path(font_path)
        self.image_size = int(image_size)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        if not self.font_path.exists():
            raise FileNotFoundError(
                f"Font not found: {self.font_path}. "
                "Place NotoSansOldSouthArabian-Regular.ttf in assets/fonts/."
            )

        self.char_to_class: Dict[str, int] = {c["char"]: int(c["id"]) for c in self.classes}
        self.letter_chars = [c["char"] for c in self.classes if c.get("type") == "letter"]
        self.separator_char = next(
            (c["char"] for c in self.classes if c.get("type") == "number_or_separator"),
            "𐩽",
        )
        self.allowed_chars = set(self.char_to_class)
        self.corpus_lines = [
            self._filter_text(line) for line in (corpus_lines or []) if self._filter_text(line)
        ]

        # Reuse a deterministic pool of stone textures. Appearance transforms
        # later in the pipeline make each use different while avoiding a large
        # per-image procedural generation cost.
        self._stone_pool = [
            _procedural_stone(self.image_size, self.image_size, self.np_rng)
            for _ in range(32)
        ]

    def _filter_text(self, text: str) -> str:
        return "".join(ch for ch in text.strip() if ch in self.allowed_chars or ch.isspace())

    def _random_line(self) -> str:
        # About one quarter of samples are intentionally short/sparse. This
        # mirrors the project requirement that a real image may contain only a
        # few characters and prevents the detector from learning that text is
        # always a long line.
        if self.rng.random() < 0.25:
            n = self.rng.randint(2, 8)
            return "".join(self.rng.choice(self.letter_chars) for _ in range(n))

        words = []
        for _ in range(self.rng.randint(1, 4)):
            n = self.rng.randint(1, 6)
            chars = [self.rng.choice(self.letter_chars) for _ in range(n)]
            words.append("".join(chars))
        return self.separator_char.join(words)

    def _choose_text(self) -> str:
        if self.corpus_lines and self.rng.random() < 0.75:
            text = self.rng.choice(self.corpus_lines)
            text = self.separator_char.join(text.split())
            text = "".join(ch for ch in text if ch in self.allowed_chars)
            if text:
                return text
        return self._random_line()

    def _background(self) -> Image.Image:
        if self.rng.random() < 0.80:
            image = self.rng.choice(self._stone_pool).copy()
            # Texture-only flips are safe because glyphs are drawn afterwards.
            if self.rng.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
            if self.rng.random() < 0.25:
                image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            return image
        return _plain_background(self.image_size, self.image_size, self.np_rng)

    def _style(self) -> dict:
        mode = self.rng.choices(
            ["dark", "incised", "faded"], weights=[0.45, 0.35, 0.20]
        )[0]
        if mode == "dark":
            ink = self.rng.randint(20, 95)
            return {
                "fill": (ink, ink, ink),
                "stroke_width": self.rng.choice([0, 0, 1]),
                "stroke_fill": None,
            }
        if mode == "incised":
            ink = self.rng.randint(45, 115)
            rim = self.rng.randint(145, 220)
            return {
                "fill": (ink, ink, ink),
                "stroke_width": self.rng.choice([1, 1, 2]),
                "stroke_fill": (rim, rim, rim),
            }
        ink = self.rng.randint(95, 155)
        return {
            "fill": (ink, ink, ink),
            "stroke_width": self.rng.choice([0, 1]),
            "stroke_fill": None,
        }

    def _render_line(
        self,
        draw: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.FreeTypeFont,
        y_top: int,
        right_margin: int,
        left_margin: int,
        spacing: int,
        style: dict,
    ) -> Tuple[List[Tuple[int, int, int, int, int]], str]:
        cursor_x = self.image_size - right_margin
        boxes: List[Tuple[int, int, int, int, int]] = []
        kept_chars: List[str] = []

        stroke_width = int(style["stroke_width"])
        for ch in text:
            if ch not in self.char_to_class:
                continue

            bbox0 = draw.textbbox((0, 0), ch, font=font, stroke_width=stroke_width)
            glyph_w = max(1, bbox0[2] - bbox0[0])
            glyph_h = max(1, bbox0[3] - bbox0[1])

            left = cursor_x - glyph_w
            top = y_top
            right = cursor_x
            bottom = y_top + glyph_h

            if left < left_margin:
                break

            kwargs = {
                "font": font,
                "fill": style["fill"],
                "stroke_width": stroke_width,
            }
            if style["stroke_fill"] is not None:
                kwargs["stroke_fill"] = style["stroke_fill"]

            draw.text(
                (left - bbox0[0], top - bbox0[1]),
                ch,
                **kwargs,
            )

            boxes.append((self.char_to_class[ch], left, top, right, bottom))
            kept_chars.append(ch)
            cursor_x = left - spacing

        return boxes, "".join(kept_chars)

    def _line_positions(self, line_count: int, glyph_height: int) -> List[int]:
        margin = max(18, self.image_size // 28)
        usable_bottom = self.image_size - margin - glyph_height
        if usable_bottom <= margin:
            return [margin]

        if line_count <= 1:
            return [self.rng.randint(margin, usable_bottom)]

        # Spread multiple lines across the canvas with small jitter instead of
        # always packing them at the top. This improves positional diversity.
        anchors = np.linspace(margin, usable_bottom, num=line_count)
        gap = max(8, int((usable_bottom - margin) / max(1, line_count - 1)))
        jitter = max(2, min(gap // 5, self.image_size // 24))
        positions = [
            int(max(margin, min(usable_bottom, round(v + self.rng.randint(-jitter, jitter)))))
            for v in anchors
        ]
        return sorted(positions)

    def make_sample(self) -> RenderedSample:
        image = self._background()
        draw = ImageDraw.Draw(image)

        font_size = self.rng.randint(38, 98)
        font = ImageFont.truetype(str(self.font_path), font_size)
        style = self._style()

        line_count = self.rng.choices([1, 2, 3, 4], weights=[0.46, 0.32, 0.17, 0.05])[0]
        # Use the tallest supported character to estimate safe vertical spacing.
        sample_bbox = draw.textbbox(
            (0, 0), "𐩣", font=font, stroke_width=int(style["stroke_width"])
        )
        glyph_height = max(1, sample_bbox[3] - sample_bbox[1])
        positions = self._line_positions(line_count, glyph_height)

        all_boxes: List[Tuple[int, int, int, int, int]] = []
        rendered_lines: List[str] = []

        for y in positions:
            text = self._choose_text()
            boxes, kept = self._render_line(
                draw,
                text,
                font,
                y,
                right_margin=self.rng.randint(24, 78),
                left_margin=self.rng.randint(24, 78),
                spacing=self.rng.randint(3, 20),
                style=style,
            )
            if boxes:
                all_boxes.extend(boxes)
                rendered_lines.append(kept)

        # Geometry-preserving degradation. Labels remain exact because these
        # operations alter appearance only, not coordinates.
        if self.rng.random() < 0.62:
            image = ImageEnhance.Contrast(image).enhance(self.rng.uniform(0.62, 1.45))
        if self.rng.random() < 0.52:
            image = ImageEnhance.Brightness(image).enhance(self.rng.uniform(0.72, 1.22))
        if self.rng.random() < 0.48:
            image = image.filter(ImageFilter.GaussianBlur(self.rng.uniform(0.10, 1.45)))

        if self.rng.random() < 0.72:
            # Low-resolution additive sensor/stone noise, upsampled before
            # compositing. This keeps generation fast for 10k+ images.
            small = max(32, self.image_size // 4)
            sigma = self.rng.uniform(2.0, 12.0)
            noise = np.clip(
                self.np_rng.normal(128.0, sigma, size=(small, small)), 0, 255
            ).astype(np.uint8)
            noise_img = Image.fromarray(noise, mode="L").resize(
                image.size, Image.Resampling.BILINEAR
            ).convert("RGB")
            image = ImageChops.add(image, noise_img, scale=1.0, offset=-128)

        labels = []
        for cls_id, x1, y1, x2, y2 in all_boxes:
            x1 = max(0, min(self.image_size - 1, x1))
            y1 = max(0, min(self.image_size - 1, y1))
            x2 = max(x1 + 1, min(self.image_size, x2))
            y2 = max(y1 + 1, min(self.image_size, y2))

            cx = ((x1 + x2) / 2) / self.image_size
            cy = ((y1 + y2) / 2) / self.image_size
            bw = (x2 - x1) / self.image_size
            bh = (y2 - y1) / self.image_size
            labels.append((cls_id, cx, cy, bw, bh))

        return RenderedSample(
            image=image,
            labels=labels,
            transcription="\n".join(rendered_lines),
        )


def load_corpus(path: str | Path | None) -> List[str]:
    if not path:
        return []
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def save_sample(
    sample: RenderedSample,
    image_path: str | Path,
    label_path: str | Path,
    transcript_path: str | Path,
) -> None:
    image_path = Path(image_path)
    label_path = Path(label_path)
    transcript_path = Path(transcript_path)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.parent.mkdir(parents=True, exist_ok=True)
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    sample.image.save(image_path, quality=92)

    lines = [
        f"{cls_id} {cx:.8f} {cy:.8f} {bw:.8f} {bh:.8f}"
        for cls_id, cx, cy, bw, bh in sample.labels
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    transcript_path.write_text(sample.transcription, encoding="utf-8")
