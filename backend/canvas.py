from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path

from PIL import Image

from .agent import Agent
from .config import (
    AGENT_RESIZE_RATIO,
    BG_COLOR,
    CANVAS_HEIGHT,
    CANVAS_WIDTH,
    FRAME_DELAY,
    FRAME_EVERY,
    MAX_DRAW_STEPS,
    MAX_OVERLAP_RATIO,
    PLACEMENT_ATTEMPTS,
)

logger = logging.getLogger(__name__)


def intersect_with_tolerance(box1, box2, max_overlap_ratio=MAX_OVERLAP_RATIO):
    # Calculate intersection box
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_width = max(0, x2 - x1)
    inter_height = max(0, y2 - y1)
    inter_area = inter_width * inter_height

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    overlap_ratio = inter_area / min(area1, area2)

    return overlap_ratio > max_overlap_ratio


class CollaborativeCanvas:
    """Shared canvas that accumulates agent drawings across inputs.

    Thread-safe: uploads can add agents while a background thread is drawing (see draw()).
    """

    def __init__(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg_color=BG_COLOR, frame_delay=FRAME_DELAY):
        self.bg_color = bg_color
        self.frame_delay = frame_delay
        self.image = Image.new("RGBA", (width, height), bg_color)
        self.agents = []
        self.used_regions = []  # to track agent locations
        self.input_counter = 0

        # Guards image, agents and used_regions; notified after every drawing step
        self._lock = threading.RLock()
        self._stepped = threading.Condition(self._lock)
        self._save_lock = threading.Lock()
        self._listeners = []  # on_frame callbacks of draw() calls still waiting
        self._drawer = None  # background drawing thread, alive while there are agents

    def add_agent(self, image: Image.Image, emotion: str = "Unknown", max_updates: int | None = None) -> Agent | None:
        """Place a new agent without too much overlap; returns it, or None when there is no room."""
        image = image.convert("RGBA")
        new_width = int(image.width * AGENT_RESIZE_RATIO)
        new_height = int(image.height * AGENT_RESIZE_RATIO)
        image = image.resize((new_width, new_height), Image.LANCZOS)

        with self._lock:
            for _ in range(PLACEMENT_ATTEMPTS):
                x = random.randint(0, self.image.width - new_width)
                y = random.randint(0, self.image.height - new_height)

                # Check overlap with other agents
                new_box = (x, y, x + new_width, y + new_height)
                overlap = any(
                    intersect_with_tolerance(new_box, region, max_overlap_ratio=MAX_OVERLAP_RATIO)
                    for region in self.used_regions
                )

                if not overlap:
                    self.used_regions.append(new_box)
                    agent = Agent(image, x, y, emotion=emotion, max_updates=max_updates or MAX_DRAW_STEPS)
                    self.agents.append(agent)
                    return agent

        logger.warning("Could not place agent without overlap.")
        return None

    def step(self) -> bool:
        """Every agent paints once and finished agents are removed; returns whether any are left."""
        with self._lock:
            for agent in self.agents:
                agent.update(self.image)
            self.agents = [agent for agent in self.agents if not agent.done]
            self._stepped.notify_all()
            return bool(self.agents)

    def run(self, steps: int | None = None, on_frame=None) -> int:
        """Draw in the calling thread until every agent is done, or for at most ``steps`` steps.

        Unfinished agents carry over to the next run. ``on_frame`` gets a flattened
        copy every ``FRAME_EVERY`` steps (each followed by a ``frame_delay`` pause)
        and once more when the run ends. Returns the number of steps taken.
        """
        with self._lock:
            self.input_counter += 1

        step = 0
        while (steps is None or step < steps) and self.agents:
            self.step()
            if on_frame and step % FRAME_EVERY == 0:
                on_frame(self.flattened())
                time.sleep(self.frame_delay)
            step += 1

        if on_frame:
            on_frame(self.flattened())  # final frame shows the finished canvas
        return step

    def draw(self, agents, on_frame=None) -> None:
        """Block until ``agents`` have finished, while a background thread draws.

        Several uploads can call this at once: they share one drawing thread, so
        every agent on the canvas paints at the same time, and agents added while
        others are drawing join in right away. ``on_frame`` gets frames (paced by
        ``frame_delay``) until this call's agents are done, then one final frame.
        """
        agents = [agent for agent in agents if agent is not None]
        with self._lock:
            self.input_counter += 1
            if on_frame:
                self._listeners.append(on_frame)
            if self._drawer is None and self.agents:
                self._drawer = threading.Thread(target=self._draw_loop, name="canvas-drawer", daemon=True)
                self._drawer.start()
            try:
                self._stepped.wait_for(lambda: not any(agent in self.agents for agent in agents))
            finally:
                if on_frame:
                    self._listeners.remove(on_frame)

        if on_frame:
            on_frame(self.flattened())  # final frame shows this upload's finished drawing

    def _draw_loop(self) -> None:
        step = 0
        while True:
            with self._lock:
                if not self.step():
                    self._drawer = None  # set under the lock, so draw() starts a new thread if needed
                    return
                # The same callback can be registered by several uploads; send each frame once
                listeners = list(dict.fromkeys(self._listeners)) if step % FRAME_EVERY == 0 else []

            if listeners:
                frame = self.flattened()
                for on_frame in listeners:
                    try:
                        on_frame(frame)
                    except Exception:
                        logger.exception("on_frame callback failed")
                time.sleep(self.frame_delay)  # outside the lock, so uploads can add agents meanwhile
            step += 1

    def flattened(self) -> Image.Image:
        # flatten image over the background colour
        with self._lock:
            background = Image.new("RGBA", self.image.size, self.bg_color)
            return Image.alpha_composite(background, self.image)

    def save(self, path) -> Path:
        path = Path(path)
        frame = self.flattened()
        with self._save_lock:  # concurrent uploads write the same file
            frame.save(path)
        logger.info("Canvas saved to %s", path)
        return path
