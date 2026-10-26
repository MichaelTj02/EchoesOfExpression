from __future__ import annotations

import logging

from PIL import Image

from .config import (
    SD_BATCH_SIZE,
    SD_GUIDANCE,
    SD_MODEL_ID,
    SD_NEGATIVE_PROMPT,
    SD_NSFW_RETRIES,
    SD_SIZE,
    SD_STEPS,
)

logger = logging.getLogger(__name__)


class ImageGenerator:
    """Stable Diffusion wrapper; the model is only loaded on load()."""

    def __init__(self, model_id: str = SD_MODEL_ID, batch_size: int = SD_BATCH_SIZE):
        self.model_id = model_id
        self.batch_size = batch_size
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
        from diffusers import AutoPipelineForText2Image

        # AutoPipeline picks the right pipeline class, so SD_MODEL_ID can point at any text-to-image model
        pipeline = AutoPipelineForText2Image.from_pretrained(
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

        # Optimizations (lower VRAM use). Attention slicing only slows CUDA down at 512x512
        if device != "cuda":
            pipeline.enable_attention_slicing()
        pipeline.enable_vae_slicing()  # decode batched images one at a time

        self.device = device
        self._pipeline = pipeline
        logger.info("Loaded %s on %s", self.model_id, device)

    def _run(self, prompts: list[str]) -> tuple[list[Image.Image], list[bool]]:
        width, height = SD_SIZE
        output = self._pipeline(
            prompts,
            negative_prompt=[SD_NEGATIVE_PROMPT] * len(prompts),
            height=height,
            width=width,
            num_inference_steps=SD_STEPS,
            guidance_scale=SD_GUIDANCE
        )
        # None when the model has no safety checker
        flagged = output.nsfw_content_detected or [False] * len(prompts)
        return output.images, list(flagged)

    def generate_batch(self, prompts: list[str]) -> list[Image.Image]:
        """One image per prompt, in order; prompts run on the GPU batch_size at a time."""
        self.load()

        images = []
        for start in range(0, len(prompts), self.batch_size):
            chunk = prompts[start:start + self.batch_size]
            chunk_images, flagged = self._run(chunk)

            # The safety checker returns black images; a new random seed usually gives a usable one
            for _ in range(SD_NSFW_RETRIES):
                retry = [i for i, bad in enumerate(flagged) if bad]
                if not retry:
                    break
                logger.info("Regenerating %d image(s) flagged by the safety checker", len(retry))
                retry_images, retry_flagged = self._run([chunk[i] for i in retry])
                for i, image, bad in zip(retry, retry_images, retry_flagged):
                    chunk_images[i], flagged[i] = image, bad

            if any(flagged):
                logger.warning("%d image(s) still blacked out by the safety checker", sum(flagged))
            images.extend(chunk_images)
        return images

    def generate(self, prompt: str) -> Image.Image:
        return self.generate_batch([prompt])[0]
