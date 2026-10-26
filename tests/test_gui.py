from pathlib import Path

import pytest

pytest.importorskip("tkinter")

from backend import PipelineResult  # noqa: E402
from gui.app import format_summary  # noqa: E402


def make_result(**overrides):
    fields = dict(
        extracted_text="Hola mundo",
        language="Spanish",
        translated_text="Hello world",
        emotion="Happiness",
        confidence=0.923,
        composition_style="Symmetrical & Balanced",
        painter="Diego Velazquez",
        visual_goal="Talavera tiles",
        prompt="A high-resolution digital painting of ...",
        generated_image_path=Path("image/generated_image.png"),
        canvas_path=Path("image/final_collaborative_canvas.png"),
    )
    fields.update(overrides)
    return PipelineResult(**fields)


def test_format_summary_lists_every_field():
    summary = format_summary(make_result())

    assert "Text: Hola mundo" in summary
    assert "Language: Spanish" in summary
    assert "Translated: Hello world" in summary
    assert "Emotion: Happiness (Confidence: 0.92)" in summary
    assert "Composition: Symmetrical & Balanced" in summary
    assert "Painter: Diego Velazquez" in summary
    assert "Cultural Visual: Talavera tiles" in summary
    assert "Prompt: A high-resolution digital painting of ..." in summary


def test_format_summary_english_input():
    summary = format_summary(make_result(language="English", translated_text=None))
    assert "Translated: Already in English" in summary


def test_format_summary_without_painter():
    assert "Painter: None suggested" in format_summary(make_result(painter=None))
