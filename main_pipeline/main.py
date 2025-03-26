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
CANVAS_WIDTH = 1024
CANVAS_HEIGHT = 768
canvas = init_canvas(CANVAS_WIDTH, CANVAS_HEIGHT)

# --- Background drawing loop thread ---
def start_drawing():
    run_live_drawing_loop(canvas, agents, steps=10000, delay=0.01)

drawing_thread = threading.Thread(target=start_drawing, daemon=True)
drawing_thread.start()

# --- Gradio input handler ---
def handle_user_text(text):
    print("🖋 Received text input, processing...")

    # Step 1: Process text to extract language + composition info
    extracted_data = process_text_input(text)

    # Step 2: Generate image & assign to an agent
    image = extracted_data["generated_image"]
    add_agent_for_image(image, canvas, agents)

    return "✅ Visual generation started. Watch the live canvas!"

# --- Gradio Interface (iPad stylus-compatible text input) ---
interface = gr.Interface(
    fn=handle_user_text,
    inputs=gr.Sketchpad(),
    outputs="text",
    title="🖋️ Write a Sentence to Visualize",
    description="Write a sentence using an iPad stylus. The system will interpret it and draw visuals based on cultural context.",
    live=True
)

interface.launch(share=True)
