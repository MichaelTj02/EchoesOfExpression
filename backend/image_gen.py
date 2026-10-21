from __future__ import annotations

import logging
import random

from PIL import Image

from .config import SD_GUIDANCE, SD_MODEL_ID, SD_SIZE_OPTIONS, SD_STEPS

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Stable Diffusion XL wrapper; the model is only loaded on load()."""

    def __init__(self, model_id: str = SD_MODEL_ID):
        self.model_id = model_id
        self.device = None
        self._pipeline = None

    @property
    def loaded(self) -> bool:
        return self._pipeline is not None

    def load(self) -> None:
        if self.loaded:
            return

        # Heavy imports kept here so importing the backend stays cheap
        import torch
        from diffusers import StableDiffusionXLPipeline

        pipeline = StableDiffusionXLPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            variant="fp16"  # use float16 weights for efficiency
        )

        # Device selection: prioritize CUDA → MPS → CPU
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        pipeline.to(device)

        # Optimizations (lower VRAM use)
        pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()

        self.device = device
        self._pipeline = pipeline
        logger.info("Loaded %s on %s", self.model_id, device)

    def generate(self, prompt: str, size: tuple[int, int] | None = None) -> Image.Image:
        self.load()

        if size is None:
            size = random.choice(SD_SIZE_OPTIONS)
        width, height = size

        return self._pipeline(
            prompt,
            height=height,
            width=width,
            num_inference_steps=SD_STEPS,
            guidance_scale=SD_GUIDANCE
        ).images[0]
