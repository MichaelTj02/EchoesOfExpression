import math
import random
from collections import deque
from PIL import Image, ImageDraw, ImageFilter

from project import EMOTION_TO_SHAPE, SHAPE_PATTERNS


class Agent:
    def __init__(self, image, start_x, start_y, emotion="Unknown", patch_size=5, input_counter=0):
        self.patch_size = patch_size
        self.origin_x = start_x
        self.origin_y = start_y
        self.visited = set()
        self.queue = deque()

        # Global fading applied once
        self.image = image.convert("RGBA")
        r, g, b, a = self.image.split()
        a = a.point(lambda p: int(p * 0.65))  # Global fade, 65% opacity
        self.image.putalpha(a)

        # Start from center
        center_x = self.origin_x + self.image.width // 2
        center_y = self.origin_y + self.image.height // 2
        self.queue.append((center_x, center_y))

        # Drawing style
        self.shape_mode = EMOTION_TO_SHAPE.get(emotion, "organic_cloud")

        self.last_seen_input = input_counter
        self.max_rounds = 4  # Track drawing rounds per agent

        self.draw_step = 0  # internal draw step tracker for shape dynamics

    def create_soft_mask(self, size, blur_radius=5, strength=0.85):
        mask = Image.new("L", (size, size), 0)
        draw = ImageDraw.Draw(mask)
        draw.ellipse((0, 0, size, size), fill=255)
        mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))
        return mask.point(lambda p: int(p * strength))

    def update(self, canvas):
        if not self.queue:
            return

        for _ in range(10):  # range to control speed of drawing, more = faster and more drawn
            if not self.queue:
                break

            x, y = self.queue.popleft()
            key = (x, y)
            if key in self.visited:
                continue
            self.visited.add(key)

            local_x = max(0, x - self.origin_x)
            local_y = max(0, y - self.origin_y)

            if (
                local_x + self.patch_size > self.image.width
                or local_y + self.patch_size > self.image.height
            ):
                continue

            patch = self.image.crop(
                (
                    local_x,
                    local_y,
                    local_x + self.patch_size,
                    local_y + self.patch_size,
                )
            )

            center_x = self.origin_x + self.image.width // 2
            center_y = self.origin_y + self.image.height // 2
            dx = x - center_x
            dy = y - center_y
            dist = math.sqrt(dx ** 2 + dy ** 2)
            max_dist = math.sqrt((self.image.width / 2) ** 2 + (self.image.height / 2) ** 2)

            # Avoid any complex result in fade formulas
            try:
                fade = SHAPE_PATTERNS.get(self.shape_mode, SHAPE_PATTERNS["organic_cloud"])(
                    dx, dy, dist, max_dist
                )
            except Exception as e:
                print(f"[Agent] Fade computation failed for shape '{self.shape_mode}': {e}")
                fade = 1.0

            fade = SHAPE_PATTERNS.get(self.shape_mode, SHAPE_PATTERNS["organic_cloud"])(
                dx, dy, dist, max_dist
            )
            mask = self.create_soft_mask(self.patch_size, blur_radius=4, strength=fade)
            canvas.paste(patch, (x, y), mask)

            for _ in range(6):
                if self.shape_mode == "spiral_form":
                    angle_offset = self.draw_step / 20  # spiral effect over time
                    angle = math.atan2(y - center_y, x - center_x) + angle_offset
                    radius = self.patch_size
                    nx = x + int(radius * math.cos(angle))
                    ny = y + int(radius * math.sin(angle))

                elif self.shape_mode == "teardrop":
                    nx = x + random.randint(-1, 1)
                    ny = y + random.randint(0, self.patch_size)

                elif self.shape_mode == "jagged_burst":
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.randint(self.patch_size, self.patch_size * 2)
                    noise = random.uniform(-4, 4)
                    nx = x + int((radius + noise) * math.cos(angle))
                    ny = y + int((radius + noise) * math.sin(angle))

                else:
                    angle = random.uniform(0, 2 * math.pi)
                    radius = random.randint(1, self.patch_size)
                    nx = x + int(radius * math.cos(angle))
                    ny = y + int(radius * math.sin(angle))

                if (nx, ny) not in self.visited:
                    if (
                        self.origin_x <= nx < self.origin_x + self.image.width - self.patch_size
                        and self.origin_y <= ny < self.origin_y + self.image.height - self.patch_size
                    ):
                        self.queue.append((nx, ny))

            # increment spiral rotation step
            self.draw_step += 1

