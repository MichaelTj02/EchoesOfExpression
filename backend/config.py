from pathlib import Path

# Canvas
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
BG_COLOR = (240, 230, 210, 255)

# Stable Diffusion 1.5 fine-tune (painterly look); SDXL was too heavy and 512x512 is enough
SD_MODEL_ID = "Lykon/dreamshaper-8"
SD_SIZE = (512, 512)  # SD 1.5's native resolution
SD_STEPS = 25
SD_GUIDANCE = 7.5
# Dreamshaper drifts towards sexualized figures, which the safety checker replaces with black images
SD_NEGATIVE_PROMPT = "nsfw, nude, naked, cleavage, lowres, blurry, text, watermark, signature"
SD_NSFW_RETRIES = 1  # regenerate images the safety checker blacked out
SD_BATCH_SIZE = 4  # prompts generated together in one pass on the GPU

# Agent placement and drawing
AGENT_RESIZE_RATIO = 0.8  # 512px images land ~410px wide on the canvas
MAX_OVERLAP_RATIO = 0.35
PLACEMENT_ATTEMPTS = 500
PATCHES_PER_UPDATE = 100  # patches each agent paints per step; a 410px painting finishes in ~4000 steps
MAX_DRAW_STEPS = 20000  # safety cap so a run always ends even if an agent never finishes
FRAME_EVERY = 5
FRAME_DELAY = 0.1  # seconds paused after each preview frame, keeps the drawing at a watchable pace

# LLM: any OpenAI-compatible server. Defaults to a local Ollama model;
# override with LLM_BASE_URL / LLM_MODEL / LLM_API_KEY in .env
LLM_BASE_URL = "http://localhost:11434/v1"
LLM_MODEL = "qwen2.5vl:7b"
LLM_WORKERS = 3  # handwriting images analyzed concurrently (Ollama also limits this via OLLAMA_NUM_PARALLEL)

# Multiple uploads: each image adds a painting, and the canvas fills up after ~15
MAX_UPLOADS_PER_BATCH = 6

# Painters the LLM suggests for the language's region; one is picked at random each upload
PAINTER_CHOICES = 3  # longer lists get padded with invented names by small local models

# Emotion labels the rest of the pipeline understands
EMOTIONS = ("Happiness", "Sadness", "Fear", "Anger", "Surprise", "Disgust")

# Output
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "image"
GENERATED_IMAGE_NAME = "generated_image.png"
FINAL_CANVAS_NAME = "final_collaborative_canvas.png"
