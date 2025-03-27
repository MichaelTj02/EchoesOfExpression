import gradio as gr
from PIL import Image
import numpy as np
import time
import cv2
import random

# Import all the functions and globals from your notebook refactored into `project.py`
from project import (
    extract_text_from_handwriting,
    translate_to_english,
    analyze_emotion,
    assign_composition_style,
    get_visual_goal,
    generate_image,
    run_live_drawing_loop,
    add_agent_for_image,
    canvas,
    agents,
    extracted_data
)

def process_and_draw(image):
    global agents, canvas, extracted_data

    generated_image_path = generate_image()
    add_agent_for_image(generated_image_path)  # ✅ Add image to agent list
    run_live_drawing_loop()  # 🌀 Trigger the drawing loop


    agents = []  # reset drawing agents

    extracted_text, detected_language = extract_text_from_handwriting(image_path)
    if not extracted_text:
        return "Failed to extract text.", None

    translated_text = translate_to_english(extracted_text) if detected_language.lower() != "english" else extracted_text
    text_analysis_data = analyze_emotion(translated_text)
    composition_style = assign_composition_style()
    visual_goal = get_visual_goal()

    extracted_data.update({
        "extracted_text": extracted_text,
        "language": detected_language,
        "translated_text": translated_text,
        "emotion": text_analysis_data.get("emotion", "Unknown"),
        "confidence": text_analysis_data.get("confidence", 0.0),
        "composition_style": composition_style,
        "visual_goal": visual_goal
    })

    image_path = generate_image()
    run_live_drawing_loop()

    return (
        f"Text: {extracted_text}\n"
        f"Language: {detected_language}\n"
        f"Translated: {translated_text if detected_language.lower() != 'english' else 'N/A'}\n"
        f"Emotion: {text_analysis_data['emotion']} (Confidence: {text_analysis_data['confidence']:.2f})\n"
        f"Composition Style: {composition_style}\n"
        f"Visual Goal: {visual_goal}",
        canvas
    )

# Gradio Interface
interface = gr.Interface(
    fn=process_and_draw,
    inputs=gr.Image(type="pil", label="Upload Handwriting Image"),
    outputs=[
        gr.Textbox(label="Analysis Output"),
        gr.Image(type="pil", label="Canvas")
    ],
    title="🖋️ Cultural AI Drawing Interface",
    allow_flagging="never"
)

if __name__ == "__main__":
    interface.launch(share=True)
