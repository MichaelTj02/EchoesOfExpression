from __future__ import annotations

import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .canvas import CollaborativeCanvas
from .config import FINAL_CANVAS_NAME, GENERATED_IMAGE_NAME, LLM_WORKERS, MAX_UPLOADS_PER_BATCH, OUTPUT_DIR
from .image_gen import ImageGenerator
from .llm import (
    analyze_emotion,
    check_connection,
    extract_text,
    fetch_cultural_info,
    get_visual_goal,
    image_to_data_url,
    make_client,
    suggest_painters,
    translate_to_english,
)
from .prompting import build_prompt, composition_for

logger = logging.getLogger(__name__)

STAGES = ("load", "ocr", "translate", "emotion", "culture", "generate", "draw", "done")


class NoTextFoundError(RuntimeError):
    pass


@dataclass
class PipelineResult:
    extracted_text: str
    language: str
    translated_text: str | None  # None when language is English
    emotion: str
    confidence: float
    composition_style: str
    painter: str | None  # None when the LLM suggested no painter
    visual_goal: str
    prompt: str
    generated_image_path: Path
    canvas_path: Path


@dataclass
class _Analysis:
    """Output of the LLM stages for one image; the painter and prompt are chosen afterwards."""

    extracted_text: str
    language: str
    translated_text: str | None
    emotion: str
    confidence: float
    composition_style: str
    visual_goal: str
    cultural_info: dict
    painters: list


def _image_label(image, position: int) -> str:
    if isinstance(image, Image.Image):
        return f"image {position}"
    return Path(image).name


def generated_image_name(number: int) -> str:
    """File name for the number-th (1-based) image generated this session, e.g. generated_image_2.png."""
    base = Path(GENERATED_IMAGE_NAME)
    return f"{base.stem}_{number}{base.suffix}"


class EchoesEngine:
    """Runs the handwriting -> emotion/culture -> image -> canvas pipeline."""

    def __init__(self, api_key=None, output_dir=OUTPUT_DIR, client=None, image_generator=None, canvas=None, rng=None):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.client = client
        self.image_generator = image_generator if image_generator is not None else ImageGenerator()
        self.canvas = canvas if canvas is not None else CollaborativeCanvas()
        self.rng = rng if rng is not None else random.Random()
        self._loaded = False
        self._load_lock = threading.Lock()
        self._generate_lock = threading.Lock()  # one Stable Diffusion call at a time on the GPU
        self._generated_count = 0

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, on_progress=None) -> None:
        with self._load_lock:  # overlapping uploads must not load the models twice
            if not self._loaded:
                self._load(on_progress or (lambda stage, msg: None))

    def _load(self, notify) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        notify("load", "Connecting to the LLM server...")
        if self.client is None:
            self.client = make_client(self.api_key)
        reply = check_connection(self.client)
        logger.info("LLM connection OK: %s", reply)

        notify("load", "Loading Stable Diffusion model...")
        self.image_generator.load()

        self._loaded = True
        notify("load", "Models loaded.")

    def process(self, image, on_progress=None, on_frame=None, draw_steps=None) -> PipelineResult:
        """Run the pipeline for one image; raises if its analysis fails."""
        result = self.process_many([image], on_progress=on_progress, on_frame=on_frame, draw_steps=draw_steps)[0]
        if isinstance(result, Exception):
            raise result
        return result

    def process_many(self, images, on_progress=None, on_frame=None, draw_steps=None) -> list[PipelineResult | Exception]:
        """Run the pipeline for several images and paint them onto the canvas at the same time.

        Returns one entry per input image, in input order: its PipelineResult, or the exception
        that stopped its analysis. Load, generation and drawing errors are raised instead.
        `on_progress` may be called from worker threads.
        """
        images = list(images)
        if len(images) > MAX_UPLOADS_PER_BATCH:
            raise ValueError(f"At most {MAX_UPLOADS_PER_BATCH} images can be uploaded at once, got {len(images)}")
        if not images:
            return []
        notify = on_progress or (lambda stage, msg: None)
        total = len(images)

        def image_notify(position, image):
            if total == 1:
                return notify  # single uploads keep unprefixed messages
            prefix = f"[{position}/{total} {_image_label(image, position)}] "
            return lambda stage, msg: notify(stage, prefix + msg)

        # Uploads may overlap: only loading and the GPU are serialized; the canvas is thread-safe
        self.load(on_progress=on_progress)

        # The LLM stages are network-bound and the client is thread-safe, so analyze concurrently
        with ThreadPoolExecutor(max_workers=min(LLM_WORKERS, total)) as executor:
            futures = [
                executor.submit(self._analyze, image, image_notify(position, image))
                for position, image in enumerate(images, start=1)
            ]
        results: list = []
        for position, (image, future) in enumerate(zip(images, futures), start=1):
            try:
                results.append(future.result())
            except Exception as e:  # one bad image must not stop the rest of the batch
                logger.warning("Analysis failed for %s: %s", _image_label(image, position), e)
                results.append(e)

        analyzed = [i for i, r in enumerate(results) if isinstance(r, _Analysis)]
        if not analyzed:
            return results

        with self._generate_lock:
            # Painters are picked here in input order, not in the workers, so a seeded rng is
            # deterministic. A different painter from the region each time, so repeat languages vary
            painters, prompts = [], []
            for i in analyzed:
                analysis = results[i]
                painter = self.rng.choice(analysis.painters) if analysis.painters else None
                painters.append(painter)
                prompts.append(build_prompt(analysis.cultural_info, analysis.composition_style, painter))

            if len(prompts) == 1:
                notify("generate", "Generating image with Stable Diffusion...")
            else:
                notify("generate", f"Generating {len(prompts)} images with Stable Diffusion...")
            generated_images = self.image_generator.generate_batch(prompts)

            # Numbered across the session, so overlapping uploads don't overwrite each other's files
            paths = []
            for generated in generated_images:
                self._generated_count += 1
                path = self.output_dir / generated_image_name(self._generated_count)
                generated.save(path)
                paths.append(path)

        notify("draw", "Drawing onto the canvas...")
        agents = []
        for i, generated in zip(analyzed, generated_images):
            agent = self.canvas.add_agent(generated, results[i].emotion, max_updates=draw_steps)
            if agent is None:
                logger.warning("Could not place agent for %s without overlap.", _image_label(images[i], i + 1))
            agents.append(agent)
        # All of this batch's paintings are drawn at the same time, alongside any earlier uploads still drawing
        self.canvas.draw(agents, on_frame=on_frame)
        canvas_path = self.canvas.save(self.output_dir / FINAL_CANVAS_NAME)

        notify("done", "Drawing complete.")

        for i, painter, prompt, path in zip(analyzed, painters, prompts, paths):
            analysis = results[i]
            results[i] = PipelineResult(
                extracted_text=analysis.extracted_text,
                language=analysis.language,
                translated_text=analysis.translated_text,
                emotion=analysis.emotion,
                confidence=analysis.confidence,
                composition_style=analysis.composition_style,
                painter=painter["name"] if painter else None,
                visual_goal=analysis.visual_goal,
                prompt=prompt,
                generated_image_path=path,
                canvas_path=canvas_path,
            )
        return results

    def _analyze(self, image, notify) -> _Analysis:
        """LLM stages for one image. Runs in a worker thread, so it must not touch shared state."""
        # Extract handwritten text & language
        notify("ocr", "Reading handwriting...")
        extracted_text, language = extract_text(self.client, image_to_data_url(image))
        if not extracted_text:
            raise NoTextFoundError("Could not extract any text from image")

        translated_text = None
        if language.lower() != "english":
            notify("translate", f"Translating from {language}...")
            translated_text = translate_to_english(self.client, extracted_text, language)

        notify("emotion", "Analyzing emotion...")
        analysis = analyze_emotion(self.client, translated_text or extracted_text)
        emotion = analysis["emotion"]

        # OCR sometimes fails to name the language; the emotion analysis also reports one
        if language == "Unknown" and analysis.get("language", "Unknown") != "Unknown":
            logger.info("OCR language unknown, falling back to analysis language: %s", analysis["language"])
            language = analysis["language"]

        notify("culture", f"Finding cultural context for {language}...")
        return _Analysis(
            extracted_text=extracted_text,
            language=language,
            translated_text=translated_text,
            emotion=emotion,
            confidence=analysis["confidence"],
            composition_style=composition_for(emotion),
            visual_goal=get_visual_goal(self.client, language),
            cultural_info=fetch_cultural_info(self.client, language),
            painters=suggest_painters(self.client, language),
        )

    def canvas_snapshot(self):
        return self.canvas.flattened()
