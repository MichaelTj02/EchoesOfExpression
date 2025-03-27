from PIL import Image
from collections import deque
import random
import time
import cv2
import numpy as np
from matplotlib import pyplot as plt

def init_canvas(width, height):
    return Image.new("RGBA", (width, height), (255, 255, 255, 0))

class Agent:
    def __init__(self, image, start_x, start_y, patch_size=5):
        self.image = image.convert("RGBA")
        self.patch_size = patch_size
        self.origin_x = start_x
        self.origin_y = start_y
        self.visited = set()
        self.queue = deque()

        self.drawing_mode = random.choice([
            "center_out", "top_down", "left_to_right", "spiral", "organic_noise"
        ])

        center_x = self.origin_x + self.image.width // 2
        center_y = self.origin_y + self.image.height // 2

        if self.drawing_mode in ["center_out", "spiral"]:
            self.queue.append((center_x, center_y))
        elif self.drawing_mode == "top_down":
            for x in range(self.origin_x, self.origin_x + self.image.width, self.patch_size):
                self.queue.append((x, self.origin_y))
        elif self.drawing_mode == "left_to_right":
            for y in range(self.origin_y, self.origin_y + self.image.height, self.patch_size):
                self.queue.append((self.origin_x, y))
        elif self.drawing_mode == "organic_noise":
            for _ in range(10):
                rx = random.randint(self.origin_x, self.origin_x + self.image.width - self.patch_size)
                ry = random.randint(self.origin_y, self.origin_y + self.image.height - self.patch_size)
                self.queue.append((rx, ry))

        print(f"🎨 Agent [{self.drawing_mode}] starting at ({start_x}, {start_y})")

    def update(self, canvas):
        if not self.queue:
            return

        x, y = self.queue.popleft()
        if (x, y) in self.visited:
            return
        self.visited.add((x, y))

        patch = self.image.crop((
            max(0, x - self.origin_x),
            max(0, y - self.origin_y),
            max(0, x - self.origin_x + self.patch_size),
            max(0, y - self.origin_y + self.patch_size)
        ))

        center_x = self.origin_x + self.image.width // 2
        center_y = self.origin_y + self.image.height // 2
        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        max_dist = ((self.image.width // 2) ** 2 + (self.image.height // 2) ** 2) ** 0.5
        fade = max(0.2, 1.0 - (dist / max_dist))

        patch = patch.copy()
        r, g, b, a = patch.split()
        a = a.point(lambda p: int(p * fade))
        patch.putalpha(a)

        canvas.paste(patch, (x, y), patch)

        directions = [
            (self.patch_size, 0), (-self.patch_size, 0),
            (0, self.patch_size), (0, -self.patch_size)
        ]

        if self.drawing_mode == "spiral":
            directions = sorted(directions, key=lambda d: random.random() + 0.3 * (d[0] + d[1]))
        elif self.drawing_mode == "top_down":
            directions = sorted(directions, key=lambda d: d[1])
        elif self.drawing_mode == "left_to_right":
            directions = sorted(directions, key=lambda d: d[0])
        elif self.drawing_mode == "organic_noise":
            random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in self.visited:
                if self.origin_x <= nx < self.origin_x + self.image.width - self.patch_size and \
                   self.origin_y <= ny < self.origin_y + self.image.height - self.patch_size:
                    self.queue.append((nx, ny))

def add_agent_for_image(image, canvas, agents_list):
    x = random.randint(0, canvas.width - image.width)
    y = random.randint(0, canvas.height - image.height)
    agent = Agent(image, x, y)
    agents_list.append(agent)

def run_live_drawing_loop(canvas, agents_list, steps=5000, delay=0.01):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    
    for step in range(steps):
        for agent in agents_list:
            agent.update(canvas)

        if step % 10 == 0:
            try:
                ax.imshow(canvas)
                ax.set_title("Live Collaborative Canvas")
                ax.axis("off")
                plt.pause(0.001)
                ax.clear()  # Clear for next frame
            except Exception as e:
                print("⚠️ Display error:", e)

        time.sleep(delay)

    canvas.save("final_collaborative_canvas.png")
    print("🖼️ Canvas saved as final_collaborative_canvas.png")
    plt.ioff()
    plt.close()