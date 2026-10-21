from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from .canvas import CollaborativeCanvas
from .config import FINAL_CANVAS_NAME, GENERATED_IMAGE_NAME, OUTPUT_DIR
from .image_gen import ImageGenerator
from .llm import (
    analyze_emotion,
    check_connection,
    extract_text,
    fetch_cultural_info,
    get_visual_goal,
    image_to_data_url,
    make_client,
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
    visual_goal: str
    prompt: str
    generated_image_path: Path
    canvas_path: Path


class EchoesEngine:
    """Runs the handwriting -> emotion/culture -> image -> canvas pipeline."""

    def __init__(self, api_key=None, output_dir=OUTPUT_DIR, client=None, image_generator=None, canvas=None):
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.client = client
        self.image_generator = image_generator if image_generator is not None else ImageGenerator()
        self.canvas = canvas if canvas is not None else CollaborativeCanvas()
        self._loaded = False
        self._lock = threading.Lock()

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, on_progress=None) -> None:
        if self._loaded:
            return
        notify = on_progress or (lambda stage, msg: None)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        notify("load", "Connecting to OpenAI...")
        if self.client is None:
            self.client = make_client(self.api_key)
        reply = check_connection(self.client)
        logger.info("OpenAI connection OK: %s", reply)

        notify("load", "Loading Stable Diffusion model...")
        self.image_generator.load()

        self._loaded = True
        notify("load", "Models loaded.")

    def process(self, image, on_progress=None, on_frame=None, draw_steps=None) -> PipelineResult:
        notify = on_progress or (lambda stage, msg: None)

        # Only one image at a time: canvas, agents and output files are shared
        with self._lock:
            if not self._loaded:
                self.load(on_progress=on_progress)

            # Extract handwritten text & language
            notify("ocr", "Reading handwriting...")
            extracted_text, language = extract_text(self.client, image_to_data_url(image))
            if not extracted_text:
                raise NoTextFoundError("Could not extract any text from image")

            translated_text = None
            if language.lower() != "english":
                notify("translate", f"Translating from {language}...")
                translated_text = translate_to_english(self.client, extracted_text)

            notify("emotion", "Analyzing emotion...")
            analysis = analyze_emotion(self.client, translated_text or extracted_text)
            emotion = analysis["emotion"]
            confidence = analysis["confidence"]

            # OCR sometimes fails to name the language; the emotion analysis also reports one
            if language == "Unknown" and analysis.get("language", "Unknown") != "Unknown":
                logger.info("OCR language unknown, falling back to analysis language: %s", analysis["language"])
                language = analysis["language"]

            notify("culture", f"Finding cultural context for {language}...")
            composition_style = composition_for(emotion)
            visual_goal = get_visual_goal(self.client, language)
            cultural_info = fetch_cultural_info(self.client, language)

            notify("generate", "Generating image with Stable Diffusion...")
            prompt = build_prompt(cultural_info, composition_style)
            generated = self.image_generator.generate(prompt)
            generated_image_path = self.output_dir / GENERATED_IMAGE_NAME
            generated.save(generated_image_path)

            notify("draw", "Drawing onto the canvas...")
            if not self.canvas.add_agent(generated, emotion):
                logger.warning("Could not place agent without overlap.")
            self.canvas.run(steps=draw_steps, on_frame=on_frame)
            canvas_path = self.canvas.save(self.output_dir / FINAL_CANVAS_NAME)
            logger.info("Canvas saved to %s", canvas_path)

            notify("done", "Drawing complete.")

            return PipelineResult(
                extracted_text=extracted_text,
                language=language,
                translated_text=translated_text,
                emotion=emotion,
                confidence=confidence,
                composition_style=composition_style,
                visual_goal=visual_goal,
                prompt=prompt,
                generated_image_path=generated_image_path,
                canvas_path=canvas_path,
            )

    def canvas_snapshot(self):
        return self.canvas.flattened()
