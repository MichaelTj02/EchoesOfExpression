import threading

import pytest
from PIL import Image

import backend.canvas as canvas_module
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


def region_changed(canvas, box):
    background = Image.new("RGBA", (box[2] - box[0], box[3] - box[1]), BG_COLOR)
    return canvas.flattened().crop(box).tobytes() != background.tobytes()


def test_run_calls_on_frame_and_draws():
    canvas = CollaborativeCanvas(width=200, height=150, frame_delay=0)
    assert canvas.add_agent(solid((100, 100)))
    before = canvas.flattened().tobytes()

    frames = []
    taken = canvas.run(steps=50, on_frame=frames.append)

    assert 0 < taken <= 50
    # one frame every FRAME_EVERY steps plus a final one for the finished canvas
    assert len(frames) == len(range(0, taken, FRAME_EVERY)) + 1
    assert all(frame.size == (200, 150) for frame in frames)
    assert frames[-1].tobytes() == canvas.flattened().tobytes()
    assert canvas.input_counter == 1
    assert canvas.flattened().tobytes() != before


def test_run_without_steps_finishes_and_removes_agents():
    canvas = CollaborativeCanvas(width=100, height=100)
    assert canvas.add_agent(solid((50, 50)))
    agent = canvas.agents[0]

    taken = canvas.run()

    assert agent.done
    assert canvas.agents == []
    assert len(canvas.used_regions) == 1  # regions stay reserved
    assert 0 < taken < canvas_module.MAX_DRAW_STEPS
    assert region_changed(canvas, canvas.used_regions[0])


def test_run_with_steps_caps_and_continues_next_run():
    canvas = CollaborativeCanvas(width=100, height=100)
    assert canvas.add_agent(solid((50, 50)))
    agent = canvas.agents[0]

    assert canvas.run(steps=2) == 2
    assert not agent.done
    assert canvas.agents == [agent]

    canvas.run()
    assert agent.done
    assert canvas.agents == []
    assert canvas.input_counter == 2


def test_run_draws_all_agents_in_one_run():
    canvas = CollaborativeCanvas(width=300, height=100)
    for emotion in ("Happiness", "Anger", "Fear"):
        assert canvas.add_agent(solid((50, 50)), emotion=emotion)

    canvas.run()

    assert canvas.agents == []
    assert canvas.input_counter == 1
    assert all(region_changed(canvas, box) for box in canvas.used_regions)


def test_run_without_agents_returns_immediately():
    canvas = CollaborativeCanvas(width=50, height=50)
    frames = []

    assert canvas.run(on_frame=frames.append) == 0
    assert len(frames) == 1  # just the final frame
    assert canvas.input_counter == 1


def test_agents_stop_at_max_draw_steps(monkeypatch):
    monkeypatch.setattr(canvas_module, "MAX_DRAW_STEPS", 3)
    canvas = CollaborativeCanvas(width=200, height=150)
    agent = canvas.add_agent(solid((100, 100)))

    assert canvas.run() == 3
    assert agent.done and agent.queue  # stopped by the cap, not because it finished
    assert canvas.agents == []


def test_add_agent_max_updates_caps_that_agent():
    canvas = CollaborativeCanvas(width=200, height=100)
    capped = canvas.add_agent(solid((50, 50)), max_updates=2)
    free = canvas.add_agent(solid((50, 50)))

    canvas.run(steps=3)

    assert capped.updates == 2 and capped not in canvas.agents
    assert canvas.agents == [free]


def test_draw_blocks_until_its_agents_finish():
    canvas = CollaborativeCanvas(width=100, height=100, frame_delay=0)
    agent = canvas.add_agent(solid((50, 50)))
    frames = []

    canvas.draw([agent], on_frame=frames.append)

    assert agent.done
    assert canvas.agents == []
    assert canvas.input_counter == 1
    assert frames and frames[-1].tobytes() == canvas.flattened().tobytes()
    assert region_changed(canvas, canvas.used_regions[0])


def test_draw_without_agents_returns_immediately():
    canvas = CollaborativeCanvas(width=50, height=50)
    frames = []

    canvas.draw([None], on_frame=frames.append)  # add_agent returns None when there is no room

    assert len(frames) == 1


def test_concurrent_draws_share_the_canvas():
    canvas = CollaborativeCanvas(width=300, height=100, frame_delay=0.001)
    first = canvas.add_agent(solid((50, 50)))
    started = threading.Event()
    frames = []

    def on_frame(frame):
        frames.append(frame)
        started.set()

    thread = threading.Thread(target=canvas.draw, args=([first],), kwargs={"on_frame": on_frame})
    thread.start()
    assert started.wait(timeout=5)

    # An upload that arrives mid-drawing joins in while the first one is still painting
    second = canvas.add_agent(solid((50, 50)), emotion="Anger")
    canvas.draw([second], on_frame=on_frame)
    thread.join(timeout=10)

    assert not thread.is_alive()
    assert first.done and second.done
    assert canvas.agents == []
    assert canvas.input_counter == 2
    assert all(region_changed(canvas, box) for box in canvas.used_regions)


def test_drawer_restarts_after_going_idle():
    canvas = CollaborativeCanvas(width=200, height=100)
    canvas.draw([canvas.add_agent(solid((50, 50)))])
    assert canvas._drawer is None

    agent = canvas.add_agent(solid((50, 50)))
    canvas.draw([agent])
    assert agent.done and canvas.agents == []


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
