import pytest
from PIL import Image

from backend.canvas import CollaborativeCanvas, intersect_with_tolerance
from backend.config import AGENT_RESIZE_RATIO, BG_COLOR, FRAME_EVERY


def solid(size, color=(200, 40, 40)):
    return Image.new("RGB", size, color)


def test_intersect_with_tolerance():
    box = (0, 0, 100, 100)
    assert intersect_with_tolerance(box, box)
    assert not intersect_with_tolerance(box, (200, 200, 300, 300))
    # 10% overlap is under the default 35% tolerance
    assert not intersect_with_tolerance(box, (90, 0, 190, 100))
    assert intersect_with_tolerance(box, (50, 0, 150, 100))


def test_new_canvas_is_filled_with_background():
    canvas = CollaborativeCanvas(width=50, height=40)
    assert canvas.image.size == (50, 40)
    assert canvas.image.mode == "RGBA"
    assert canvas.image.getpixel((10, 10)) == BG_COLOR
    assert canvas.agents == []
    assert canvas.used_regions == []
    assert canvas.input_counter == 0


def test_add_agent_places_resized_image_inside_canvas():
    canvas = CollaborativeCanvas(width=400, height=300)
    assert canvas.add_agent(solid((200, 150)), emotion="Sadness")

    assert len(canvas.agents) == 1
    assert len(canvas.used_regions) == 1
    x1, y1, x2, y2 = canvas.used_regions[0]
    assert (x2 - x1, y2 - y1) == (int(200 * AGENT_RESIZE_RATIO), int(150 * AGENT_RESIZE_RATIO))
    assert 0 <= x1 and x2 <= 400 and 0 <= y1 and y2 <= 300
    assert canvas.agents[0].shape_mode == "teardrop"


def test_add_agent_rejects_overlapping_placement():
    side = int(120 / AGENT_RESIZE_RATIO)  # resizes to (at most) the full canvas
    canvas = CollaborativeCanvas(width=120, height=120)

    assert canvas.add_agent(solid((side, side)))
    assert not canvas.add_agent(solid((side, side)))
    assert len(canvas.agents) == 1
    assert len(canvas.used_regions) == 1


def test_run_calls_on_frame_and_draws():
    canvas = CollaborativeCanvas(width=200, height=150)
    assert canvas.add_agent(solid((100, 100)))
    before = canvas.flattened().tobytes()

    frames = []
    canvas.run(steps=50, on_frame=frames.append)

    assert len(frames) == len(range(0, 50, FRAME_EVERY))
    assert all(frame.size == (200, 150) for frame in frames)
    assert canvas.input_counter == 1
    assert canvas.flattened().tobytes() != before


def test_run_without_steps_uses_random_range(monkeypatch):
    import backend.canvas as canvas_module

    monkeypatch.setattr(canvas_module.random, "randint", lambda a, b: 3)
    canvas = CollaborativeCanvas(width=50, height=50)
    frames = []
    canvas.run(on_frame=frames.append)
    assert len(frames) == len(range(0, 3, FRAME_EVERY))


def test_agents_are_pruned_after_max_rounds():
    canvas = CollaborativeCanvas(width=200, height=150)
    assert canvas.add_agent(solid((100, 100)))
    max_rounds = canvas.agents[0].max_rounds

    for _ in range(max_rounds):
        canvas.run(steps=1)
    assert len(canvas.agents) == 1

    canvas.run(steps=1)
    assert canvas.agents == []


def test_flattened_returns_a_new_image():
    canvas = CollaborativeCanvas(width=30, height=20)
    flat = canvas.flattened()

    assert flat is not canvas.image
    assert flat.size == canvas.image.size
    flat.putpixel((0, 0), (0, 0, 0, 255))
    assert canvas.image.getpixel((0, 0)) == BG_COLOR


def test_save_writes_file(tmp_path):
    canvas = CollaborativeCanvas(width=30, height=20)
    path = canvas.save(tmp_path / "canvas.png")

    assert path == tmp_path / "canvas.png"
    with Image.open(path) as saved:
        assert saved.size == (30, 20)
