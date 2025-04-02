import gradio as gr
import threading
import os
from dotenv import load_dotenv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import json

# --- Load environment variables ---
load_dotenv()

# --- Load OpenAI key ---
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

# --- Import your own modules ---
from canvas_utils import run_live_drawing_loop, add_agent_for_image, init_canvas
from generation_pipeline import process_text_input

# --- Initialize the canvas and agent list ---
CANVAS_WIDTH = 2560
CANVAS_HEIGHT = 1440
canvas = init_canvas(CANVAS_WIDTH, CANVAS_HEIGHT)
agents = []

# --- Background drawing loop thread ---
def start_drawing_loop():
    run_live_drawing_loop(canvas, agents, steps=10000, delay=0.01)

# drawing_thread = threading.Thread(target=start_drawing_loop, daemon=True)
# drawing_thread.start()

# --- Gradio input handler ---
def handle_user_input(sketch_image):
    print("📥 Incoming Gradio sketch_image:", type(sketch_image))
    if isinstance(sketch_image, dict):
        print("📦 Dict keys:", sketch_image.keys())

    result = process_text_input(sketch_image)

    # 🧠 Add the generated image to the canvas
    generated_img = result.get("generated_image")
    if generated_img:
        add_agent_for_image(generated_img, canvas, agents)

    # 🖼️ Return both image + text to Gradio
    return generated_img, json.dumps(result["extracted_data"], indent=2)

def start_drawing_live_matplotlib():
    fig, ax = plt.subplots()
    ax.set_title("Live Collaborative Canvas")
    im_display = ax.imshow(canvas)
    plt.axis("off")

    def update(_):
        for agent in agents:
            agent.update(canvas)
        im_display.set_data(canvas)
        return [im_display]

    ani = FuncAnimation(fig, update, interval=50, blit=True)
    plt.show()

# --- Launch Gradio interface ---
interface = gr.Interface(
    fn=handle_user_input,
    inputs=gr.Sketchpad(),
    outputs=[
        gr.Image(type="pil", label="Generated Image"),
        gr.Textbox(label="Extracted Data")
    ],
    title="Cultural AI Sketch Interface"
)

# if __name__ == "__main__":
#     # Launch canvas loop in background thread
#     drawing_thread = threading.Thread(target=start_drawing, daemon=True)
#     drawing_thread.start()

#     # Run Gradio on main thread (this blocks)
#     interface.launch(share=True)
    
if __name__ == "__main__":
    # Run Gradio first so it doesn't get blocked
    threading.Thread(target=interface.launch, kwargs={"share": True}, daemon=True).start()

    # Run the canvas display in the main thread safely
    start_drawing_live_matplotlib()
