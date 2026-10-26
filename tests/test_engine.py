import random
import threading
import time
from pathlib import Path

import pytest
from PIL import Image

from backend import EchoesEngine, NoTextFoundError, PipelineResult
from backend.canvas import CollaborativeCanvas
from backend.config import FINAL_CANVAS_NAME, GENERATED_IMAGE_NAME, LLM_WORKERS, MAX_UPLOADS_PER_BATCH
from backend.engine import STAGES
from backend.llm import image_to_data_url
from conftest import OCR, FakeOpenAIClient, StubImageGenerator, _response, make_fake_client


def make_engine(tmp_path, client=None, generator=None, rng=None):
    return EchoesEngine(
        output_dir=tmp_path / "out",
        client=client or make_fake_client(),
        image_generator=generator or StubImageGenerator(),
        canvas=CollaborativeCanvas(width=400, height=300),
        rng=rng,
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


def run_many(engine, images, **kwargs):
    events = []
    frames = []
    results = engine.process_many(
        images,
        on_progress=lambda stage, message: events.append((stage, message)),
        on_frame=frames.append,
        draw_steps=10,
        **kwargs,
    )
    return results, events, frames


def generated_name(position):
    base = Path(GENERATED_IMAGE_NAME)
    return f"{base.stem}_{position}{base.suffix}"


def _image_url(kwargs):
    for message in kwargs.get("messages", []):
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return item["image_url"]["url"]
    return None


class PerImageClient(FakeOpenAIClient):
    """Fake client whose OCR answer depends on which image is sent.

    `ocr` maps an image's data URL to a response string, an exception to raise,
    or a callable that runs before the default OCR response is returned.
    """

    def __init__(self, ocr, **fake_kwargs):
        super().__init__(make_fake_client(**fake_kwargs).responses)
        self.ocr = ocr

    def _create(self, **kwargs):
        value = self.ocr.get(_image_url(kwargs))
        if value is None:
            return super()._create(**kwargs)
        self.calls.append(kwargs)
        if isinstance(value, Exception):
            raise value
        if callable(value):
            value()
            return _response(self.responses[OCR])
        return _response(value)


def ocr_reply(text):
    return f'{{"text": "{text}", "language": "English"}}'


@pytest.fixture
def handwriting_pngs(tmp_path):
    """Distinct handwriting images (different pixels, so different data URLs)."""
    paths = []
    for i in range(1, MAX_UPLOADS_PER_BATCH + 2):
        path = tmp_path / f"handwriting{i}.png"
        Image.new("RGB", (64, 32), (255, 255 - i, 255)).save(path)
        paths.append(path)
    return paths


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
    assert result.painter == "Raden Saleh"
    assert "in the style of Raden Saleh, romantic, dramatic light" in result.prompt
    assert generator.prompts == [result.prompt]
    assert frames and all(isinstance(frame, Image.Image) for frame in frames)
    assert not any(message.startswith("[") for _, message in events)  # single uploads are unprefixed


def test_process_writes_output_files(tmp_path, handwriting_png):
    engine = make_engine(tmp_path)
    result, _, _ = run(engine, handwriting_png)

    out = tmp_path / "out"
    assert result.generated_image_path == out / generated_name(1)
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


def test_process_picks_painter_at_random(tmp_path, handwriting_png):
    client = make_fake_client(
        painters='{"painters": [{"name": "Claude Monet", "style": "a"}, {"name": "Henri Matisse", "style": "b"}, '
                 '{"name": "Paul Cezanne", "style": "c"}]}'
    )
    engine = make_engine(tmp_path, client, rng=random.Random(0))
    expected = random.Random(0).choice(["Claude Monet", "Henri Matisse", "Paul Cezanne"])

    result, _, _ = run(engine, handwriting_png)

    assert result.painter == expected


def test_process_without_painters_uses_default_style(tmp_path, handwriting_png):
    engine = make_engine(tmp_path, make_fake_client(painters="not json"))
    result, _, _ = run(engine, handwriting_png)
    assert result.painter is None
    assert "expressive textured brushwork" in result.prompt


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
    assert len(engine.canvas.used_regions) == 2


def test_canvas_snapshot_is_a_copy(tmp_path):
    engine = make_engine(tmp_path)
    snapshot = engine.canvas_snapshot()

    assert snapshot is not engine.canvas.image
    assert snapshot.size == (400, 300)


def test_process_many_returns_results_in_input_order(tmp_path, handwriting_pngs):
    paths = handwriting_pngs[:3]
    ocr = {image_to_data_url(path): ocr_reply(f"Text {i}") for i, path in enumerate(paths, start=1)}
    # The first image answers last, so completion order differs from input order
    ocr[image_to_data_url(paths[0])] = lambda: time.sleep(0.2)
    engine = make_engine(tmp_path, PerImageClient(ocr, text="Text 1"))

    results, _, _ = run_many(engine, paths)

    assert all(isinstance(result, PipelineResult) for result in results)
    assert [result.extracted_text for result in results] == ["Text 1", "Text 2", "Text 3"]


def test_process_many_failure_does_not_stop_other_images(tmp_path, handwriting_pngs):
    paths = handwriting_pngs[:3]
    client = PerImageClient({image_to_data_url(paths[1]): RuntimeError("vision failed")})
    generator = StubImageGenerator()
    engine = make_engine(tmp_path, client, generator)

    results, _, _ = run_many(engine, paths)

    assert isinstance(results[0], PipelineResult)
    assert isinstance(results[1], NoTextFoundError)
    assert isinstance(results[2], PipelineResult)
    assert generator.batches == [[results[0].prompt, results[2].prompt]]
    assert len(engine.canvas.used_regions) == 2


def test_process_many_generates_all_prompts_in_one_batch(tmp_path, handwriting_pngs):
    generator = StubImageGenerator()
    engine = make_engine(tmp_path, generator=generator)

    results, events, _ = run_many(engine, handwriting_pngs[:3])

    assert generator.batches == [[result.prompt for result in results]]
    assert [stage for stage, _ in events].count("generate") == 1


def test_process_many_runs_canvas_once(tmp_path, handwriting_pngs, monkeypatch):
    engine = make_engine(tmp_path)
    agents_at_run = []
    original_run = engine.canvas.run

    def counting_run(*args, **kwargs):
        agents_at_run.append(len(engine.canvas.agents))
        return original_run(*args, **kwargs)

    monkeypatch.setattr(engine.canvas, "run", counting_run)

    run_many(engine, handwriting_pngs[:3])

    assert agents_at_run == [3]  # one run, with every new agent already placed
    assert engine.canvas.input_counter == 1


def test_process_many_analyzes_concurrently(tmp_path, handwriting_pngs):
    n = min(LLM_WORKERS, MAX_UPLOADS_PER_BATCH)
    assert n >= 2
    paths = handwriting_pngs[:n]
    # Each OCR call waits until all n are in flight; run one at a time, the barrier would time out
    barrier = threading.Barrier(n, timeout=5)
    engine = make_engine(tmp_path, PerImageClient({image_to_data_url(path): barrier.wait for path in paths}))

    results, _, _ = run_many(engine, paths)

    assert all(isinstance(result, PipelineResult) for result in results), results


def test_process_many_rejects_too_many_images(tmp_path, handwriting_pngs):
    generator = StubImageGenerator()
    engine = make_engine(tmp_path, generator=generator)

    with pytest.raises(ValueError):
        engine.process_many(handwriting_pngs[:MAX_UPLOADS_PER_BATCH + 1])

    assert not engine.loaded
    assert generator.batches == []


def test_process_many_all_failed_skips_canvas(tmp_path, handwriting_pngs):
    client = make_fake_client()
    client.responses[OCR] = RuntimeError("vision failed")
    generator = StubImageGenerator()
    engine = make_engine(tmp_path, client, generator)

    results, events, frames = run_many(engine, handwriting_pngs[:2])

    assert all(isinstance(result, NoTextFoundError) for result in results)
    assert generator.batches == []
    assert engine.canvas.input_counter == 0
    assert frames == []
    assert not {"generate", "draw", "done"} & {stage for stage, _ in events}
    assert not (tmp_path / "out" / FINAL_CANVAS_NAME).exists()


def test_process_many_writes_one_generated_image_per_upload(tmp_path, handwriting_pngs):
    paths = handwriting_pngs[:3]
    engine = make_engine(tmp_path, PerImageClient({image_to_data_url(paths[1]): RuntimeError("vision failed")}))

    results, _, _ = run_many(engine, paths)

    out = tmp_path / "out"
    assert results[0].generated_image_path == out / generated_name(1)
    assert results[2].generated_image_path == out / generated_name(3)
    assert (out / generated_name(1)).is_file()
    assert not (out / generated_name(2)).exists()  # its analysis failed
    assert (out / generated_name(3)).is_file()
    assert results[0].canvas_path == results[2].canvas_path == out / FINAL_CANVAS_NAME


def test_process_many_prefixes_per_image_progress(tmp_path, handwriting_pngs):
    engine = make_engine(tmp_path)

    _, events, _ = run_many(engine, [handwriting_pngs[0], Image.new("RGB", (64, 32), (255, 255, 255))])

    per_image = [message for stage, message in events if stage in ("ocr", "translate", "emotion", "culture")]
    assert sum(message.startswith("[1/2 handwriting1.png] ") for message in per_image) == 3
    assert sum(message.startswith("[2/2 image 2] ") for message in per_image) == 3
    assert len(per_image) == 6
    batch_wide = [message for stage, message in events if stage in ("generate", "draw", "done")]
    assert batch_wide and not any(message.startswith("[") for message in batch_wide)


def test_process_many_picks_painters_deterministically(tmp_path, handwriting_pngs):
    names = ["Claude Monet", "Henri Matisse", "Paul Cezanne"]
    client = make_fake_client(
        painters='{"painters": [{"name": "Claude Monet", "style": "a"}, {"name": "Henri Matisse", "style": "b"}, '
                 '{"name": "Paul Cezanne", "style": "c"}]}'
    )
    engine = make_engine(tmp_path, client, rng=random.Random(0))
    rng = random.Random(0)
    expected = [rng.choice(names) for _ in range(3)]

    results, _, _ = run_many(engine, handwriting_pngs[:3])

    assert [result.painter for result in results] == expected
