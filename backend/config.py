from pathlib import Path

# Canvas
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
BG_COLOR = (240, 230, 210, 255)

# Stable Diffusion 2.1
SD_MODEL_ID = "stabilityai/stable-diffusion-2-1"
SD_SIZE_OPTIONS = [
    (768, 512), (512, 768), (640, 640), (768, 576),
    (576, 768), (704, 512), (512, 704), (640, 480)
]
SD_STEPS = 30  # slightly higher for better quality
SD_GUIDANCE = 8.0

# Agent placement and drawing
AGENT_RESIZE_RATIO = 0.6  # Scale down to 60% of original image
MAX_OVERLAP_RATIO = 0.35
PLACEMENT_ATTEMPTS = 500
DRAW_STEPS_RANGE = (3000, 4500)
FRAME_EVERY = 5
FRAME_DELAY = 0.1  # seconds paused after each preview frame, keeps the drawing at a watchable pace

# OpenAI
LLM_MODEL = "gpt-4o"

# Output
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "image"
GENERATED_IMAGE_NAME = "generated_image.png"
FINAL_CANVAS_NAME = "final_collaborative_canvas.png"
