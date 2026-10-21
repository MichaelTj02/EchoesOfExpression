import pytest
from PIL import Image

from backend import EchoesEngine, NoTextFoundError, PipelineResult
from backend.canvas import CollaborativeCanvas
from backend.config import FINAL_CANVAS_NAME, GENERATED_IMAGE_NAME
from backend.engine import STAGES
from conftest import OCR, FakeOpenAIClient, StubImageGenerator, make_fake_client


def make_engine(tmp_path, client=None, generator=None):
    return EchoesEngine(
        output_dir=tmp_path / "out",
        client=client or make_fake_client(),
        image_generator=generator or StubImageGenerator(),
        canvas=CollaborativeCanvas(width=400, height=300),
    )


def run(engine, image, **kwargs):
    events = []
    frames = []
    result = engine.process(
        image,
        on_progress=lambda stage, message: events.append((stage, message)),
        on_frame=frames.append,
        draw_steps=10,
        **kwargs,
    )
    return result, events, frames


def stage_order(events):
    order = []
    for stage, _ in events:
        assert stage in STAGES
        if not order or order[-1] != stage:
            order.append(stage)
    return [stage for stage in order if stage != "load"]


def test_init_does_no_io(tmp_path):
    engine = EchoesEngine(output_dir=tmp_path / "out")
    assert not engine.loaded
    assert not (tmp_path / "out").exists()


def test_load_is_idempotent_and_creates_output_dir(tmp_path):
    client = make_fake_client()
    generator = StubImageGenerator()
    engine = make_engine(tmp_path, client, generator)

    events = []
    engine.load(on_progress=lambda stage, message: events.append(stage))
    engine.load()

    assert engine.loaded
    assert (tmp_path / "out").is_dir()
    assert generator.load_count == 1
    assert len(client.calls) == 1  # one connection check
    assert events and set(events) == {"load"}


def test_process_english(tmp_path, handwriting_png):
    generator = StubImageGenerator()
    engine = make_engine(tmp_path, generator=generator)

    result, events, frames = run(engine, handwriting_png)

    assert isinstance(result, PipelineResult)
    assert stage_order(events) == ["ocr", "emotion", "culture", "generate", "draw", "done"]
    assert result.extracted_text == "Hello world"
    assert result.language == "English"
    assert result.translated_text is None
    assert result.emotion == "Happiness"
    assert result.confidence == pytest.approx(0.9)
    assert result.composition_style == "Symmetrical & Balanced"
    assert result.visual_goal == "Mandala patterns"
    for value in ("Balinese", "batik painting", "lotus flowers", "Balinese script", "Symmetrical & Balanced"):
        assert value in result.prompt
    assert generator.prompts == [result.prompt]
    assert frames and all(isinstance(frame, Image.Image) for frame in frames)


def test_process_writes_output_files(tmp_path, handwriting_png):
    engine = make_engine(tmp_path)
    result, _, _ = run(engine, handwriting_png)

    out = tmp_path / "out"
    assert result.generated_image_path == out / GENERATED_IMAGE_NAME
    assert result.canvas_path == out / FINAL_CANVAS_NAME
    assert result.generated_image_path.is_file()
    assert result.canvas_path.is_file()
    with Image.open(result.canvas_path) as canvas:
        assert canvas.size == (400, 300)


def test_process_non_english_translates(tmp_path, handwriting_png):
    client = make_fake_client(
        text="Hola mundo",
        language="Spanish",
        translation="Hello world",
        emotion="Sadness",
        analysis_language="English",
    )
    engine = make_engine(tmp_path, client)

    result, events, _ = run(engine, handwriting_png)

    assert stage_order(events) == ["ocr", "translate", "emotion", "culture", "generate", "draw", "done"]
    assert result.extracted_text == "Hola mundo"
    assert result.language == "Spanish"
    assert result.translated_text == "Hello world"
    assert result.emotion == "Sadness"
    assert result.composition_style == "Soft, Scattered & Faded"


def test_process_falls_back_to_analysis_language(tmp_path, handwriting_png):
    client = make_fake_client(
        text="Bonjour",
        language="Unknown",
        translation="Hello",
        analysis_language="French",
    )
    engine = make_engine(tmp_path, client)

    result, _, _ = run(engine, handwriting_png)

    assert result.language == "French"
    assert any("French" in prompt and "cultural visual designer" in prompt for prompt in client.prompts())


def test_process_accepts_pil_image(tmp_path):
    engine = make_engine(tmp_path)
    result, _, _ = run(engine, Image.new("RGB", (64, 32), (255, 255, 255)))

    assert result.extracted_text == "Hello world"
    assert not (tmp_path / "uploaded_image.jpg").exists()


def test_process_loads_lazily(tmp_path, handwriting_png):
    generator = StubImageGenerator()
    engine = make_engine(tmp_path, generator=generator)
    assert not engine.loaded

    run(engine, handwriting_png)

    assert engine.loaded
    assert generator.load_count == 1


def test_process_raises_when_no_text(tmp_path, handwriting_png):
    client = make_fake_client()
    client.responses[OCR] = RuntimeError("vision failed")
    engine = make_engine(tmp_path, client)

    with pytest.raises(NoTextFoundError):
        run(engine, handwriting_png)

    # The lock must be released so the next upload still works
    client.responses[OCR] = '{"text": "Second try", "language": "English"}'
    result, _, _ = run(engine, handwriting_png)
    assert result.extracted_text == "Second try"


def test_canvas_accumulates_across_runs(tmp_path, handwriting_png):
    engine = make_engine(tmp_path)

    run(engine, handwriting_png)
    run(engine, handwriting_png)

    assert engine.canvas.input_counter == 2
    assert len(engine.canvas.agents) == 2


def test_canvas_snapshot_is_a_copy(tmp_path):
    engine = make_engine(tmp_path)
    snapshot = engine.canvas_snapshot()

    assert snapshot is not engine.canvas.image
    assert snapshot.size == (400, 300)
