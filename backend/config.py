from pathlib import Path

# Canvas
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
BG_COLOR = (240, 230, 210, 255)

# Stable Diffusion XL (SD 2.1 was removed from Hugging Face by Stability AI)
SD_MODEL_ID = "stabilityai/stable-diffusion-xl-base-1.0"
# SDXL's native ~1 megapixel resolutions: landscape, portrait and square
SD_SIZE_OPTIONS = [
    (1152, 896), (896, 1152), (1024, 1024), (1216, 832),
    (832, 1216), (1344, 768), (768, 1344), (1152, 960)
]
SD_STEPS = 30  # slightly higher for better quality
SD_GUIDANCE = 8.0

# Agent placement and drawing
AGENT_RESIZE_RATIO = 0.4  # Scale down to 40%; keeps SDXL images about the size SD 2.1 images were at 60%
MAX_OVERLAP_RATIO = 0.35
PLACEMENT_ATTEMPTS = 500
DRAW_STEPS_RANGE = (3000, 4500)
FRAME_EVERY = 5
FRAME_DELAY = 0.1  # seconds paused after each preview frame, keeps the drawing at a watchable pace

# LLM: any OpenAI-compatible server. Defaults to a local Ollama model;
# override with LLM_BASE_URL / LLM_MODEL / LLM_API_KEY in .env
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5vl:7b"

# Painters the LLM suggests for the language's region; one is picked at random each upload
PAINTER_CHOICES = 3  # longer lists get padded with invented names by small local models

# Emotion labels the rest of the pipeline understands
EMOTIONS = ("Happiness", "Sadness", "Fear", "Anger", "Surprise", "Disgust")

# Output
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "image"
GENERATED_IMAGE_NAME = "generated_image.png"
FINAL_CANVAS_NAME = "final_collaborative_canvas.png"
