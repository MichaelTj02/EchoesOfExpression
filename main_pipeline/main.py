import gradio as gr
import threading
import os
import time
from dotenv import load_dotenv
from PIL import Image

# --- Load environment variables ---
load_dotenv()

# --- Load OpenAI key ---
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

# --- Import your own modules ---
from canvas_utils import run_live_drawing_loop, add_agent_for_image, init_canvas, agents
from generation_pipeline import process_text_input

# --- Initialize the canvas (global) ---
CANVAS_WIDTH = 2560
CANVAS_HEIGHT = 1440
canvas = init_canvas(CANVAS_WIDTH, CANVAS_HEIGHT)

# --- Background drawing loop thread ---
def start_drawing():
    run_live_drawing_loop(canvas, agents, steps=10000, delay=0.01)

drawing_thread = threading.Thread(target=start_drawing, daemon=True)
drawing_thread.start()

# --- Gradio input handler ---
def handle_user_input(sketch_image):
    print("📥 Incoming Gradio sketch_image:", type(sketch_image))
    if isinstance(sketch_image, dict):
        print("📦 Dict keys:", sketch_image.keys())
    return process_text_input(sketch_image)

interface = gr.Interface(
    fn=handle_user_input,
    inputs=gr.Sketchpad(),
    outputs="text",
    title="Cultural AI Sketch Interface"
)

if __name__ == "__main__":
    import threading
    from canvas_utils import run_live_drawing_loop

    threading.Thread(target=interface.launch, kwargs={"share": True}).start()
    run_live_drawing_loop()

interface.launch(share=True)
