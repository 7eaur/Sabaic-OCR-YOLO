from __future__ import annotations

import random
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


class PhotometricAugment:
    """Geometry-preserving augmentation; horizontal mirroring is intentionally avoided."""
    def __init__(self, brightness: float = 0.25, contrast: float = 0.30, blur_probability: float = 0.20, noise_probability: float = 0.25, max_noise_std: float = 10.0):
        self.brightness = brightness
        self.contrast = contrast
        self.blur_probability = blur_probability
        self.noise_probability = noise_probability
        self.max_noise_std = max_noise_std

    def __call__(self, image: Image.Image) -> Image.Image:
        if self.brightness > 0:
            image = ImageEnhance.Brightness(image).enhance(random.uniform(1-self.brightness, 1+self.brightness))
        if self.contrast > 0:
            image = ImageEnhance.Contrast(image).enhance(random.uniform(1-self.contrast, 1+self.contrast))
        if random.random() < self.blur_probability:
            image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 1.2)))
        if random.random() < self.noise_probability:
            arr = np.asarray(image).astype(np.float32)
            arr += np.random.normal(0.0, random.uniform(1.0, self.max_noise_std), size=arr.shape)
            image = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), mode="RGB")
        return image
