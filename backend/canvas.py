from __future__ import annotations

import logging
import random
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
    """Shared canvas that accumulates agent drawings across inputs."""

    def __init__(self, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg_color=BG_COLOR, frame_delay=FRAME_DELAY):
        self.bg_color = bg_color
        self.frame_delay = frame_delay
        self.image = Image.new("RGBA", (width, height), bg_color)
        self.agents = []
        self.used_regions = []  # to track agent locations
        self.input_counter = 0

    def add_agent(self, image: Image.Image, emotion: str = "Unknown") -> bool:
        image = image.convert("RGBA")
        new_width = int(image.width * AGENT_RESIZE_RATIO)
        new_height = int(image.height * AGENT_RESIZE_RATIO)
        image = image.resize((new_width, new_height), Image.LANCZOS)

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
                self.agents.append(Agent(image, x, y, emotion=emotion))
                return True

        logger.warning("Could not place agent without overlap.")
        return False

    def run(self, steps: int | None = None, on_frame=None) -> int:
        """Let every agent paint until it has finished its drawing.

        With ``steps=None`` the run lasts until all agents are done (capped at
        ``MAX_DRAW_STEPS``); with ``steps=N`` it lasts at most N steps. Finished
        agents are removed; unfinished ones carry over to the next run.
        ``on_frame`` gets a flattened copy every ``FRAME_EVERY`` steps (each
        followed by a ``frame_delay`` pause) and once more when the run ends.
        Returns the number of steps taken.
        """
        self.input_counter += 1
        limit = MAX_DRAW_STEPS if steps is None else steps

        step = 0
        while step < limit and self.agents:
            for agent in self.agents:
                agent.update(self.image)
            self.agents = [agent for agent in self.agents if not agent.done]

            if on_frame and step % FRAME_EVERY == 0:
                on_frame(self.flattened())
                time.sleep(self.frame_delay)
            step += 1

        if steps is None and self.agents:
            logger.warning(
                "Stopped drawing after %d steps with %d unfinished agent(s).", step, len(self.agents)
            )
        if on_frame:
            on_frame(self.flattened())  # final frame shows the finished canvas
        return step

    def flattened(self) -> Image.Image:
        # flatten image over the background colour
        background = Image.new("RGBA", self.image.size, self.bg_color)
        return Image.alpha_composite(background, self.image)

    def save(self, path) -> Path:
        path = Path(path)
        self.flattened().save(path)
        logger.info("Canvas saved to %s", path)
        return path
