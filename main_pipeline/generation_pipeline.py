import os
import re
import json
import base64
import openai
import cv2
import torch
from PIL import Image
from dotenv import load_dotenv
from diffusers import StableDiffusionPipeline
from canvas_utils import add_agent_for_image

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
client = openai.OpenAI(api_key=openai.api_key)

# Stable Diffusion 2.1
MODEL_ID = "stabilityai/stable-diffusion-2-1"
pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    revision="fp16"
)
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipeline.to(device)
pipeline.safety_checker = None
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()

# Composition mapping
COMPOSITION_MAPPING = {
    "Happiness": "Symmetrical & Balanced",
    "Sadness": "Soft, Scattered & Faded",
    "Anger": "Chaotic & Overlapping",
    "Fear": "Dark & Enclosed",
    "Surprise": "Expanding & Explosive",
    "Disgust": "Distorted & Melting",
    "Unknown": "Abstract Freeform"
}

# Preprocess handwriting image
def preprocess_handwriting(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    processed = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )
    processed_path = "processed_handwriting.jpg"
    cv2.imwrite(processed_path, processed)
    return processed_path

# Convert image to Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# Extract handwriting image
def extract_text_from_handwriting(image_path):
    base64_image = image_to_base64(image_path)
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":
                            "Extract the handwritten text from this image and detect its language."
                            "Return the result in JSON format: {\"text\": \"extracted text\", \"language\": \"detected language\"}."
                            "Do not add extra explanations, just return valid JSON."
                        },
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=500
        )
        raw_response = response.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)
        else:
            raise ValueError("Invalid JSON format received from GPT.")
        text_data = json.loads(json_text)
        return text_data.get("text", "Unknown"), text_data.get("language", "Unknown")
    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None, "Unknown"

def translate_to_english(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Detect the language of this text: '{text}'. If it's not English, translate it to English. If it's already in English, return the same text."}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Translation Error: {e}")
        return text

def analyze_emotion(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Analyze the following text: '{text}'."
                 " Return ONLY a JSON object with the following structure:"
                 " {{\"emotion\": \"Happiness\", \"confidence\": 0.92, \"language\": \"English\"}}."
                 " The 'emotion' should be one of: Happiness, Sadness, Fear, Anger, Surprise, or Disgust."
                 " No explanations, no extra text—return only a valid JSON output."}
            ],
            max_tokens=100
        )
        raw_response = response.choices[0].message.content.strip()
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(0))
    except Exception as e:
        print(f"Text Analysis Error: {e}")
    return {"emotion": "Unknown", "confidence": 0.0, "language": "Unknown"}

def assign_composition_style(emotion):
    return COMPOSITION_MAPPING.get(emotion, COMPOSITION_MAPPING["Unknown"])

def fetch_cultural_prompt_info(language):
    if language == "Unknown":
        return {
            "culture": language,
            "art_form": "traditional art",
            "motif": "cultural motif",
            "script": f"{language} script"
        }
    system_prompt = (
        "You are a cultural visual designer. Given a language, respond ONLY in raw JSON format with these 4 keys:\n"
        "'culture', 'art_form', 'motif', 'script'.\n"
        "Keep each value short (2–6 words). DO NOT return markdown or explanations."
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"The language is: {language}."}
            ],
            max_tokens=150
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.strip("`").strip()
            if content.startswith("json"):
                content = content[4:].strip()
        return json.loads(content)
    except Exception as e:
        print(f"GPT API Error: {e}")
        return {
            "culture": language,
            "art_form": "traditional art",
            "motif": "cultural motif",
            "script": f"{language} script"
        }

def build_prompt(data):
    return (
        f"A high-resolution digital painting of a symbolic {data['culture']} scene or object, "
        f"inspired by traditional {data['art_form']}, "
        f"featuring {data['motif']}, layered with {data['script']}, "
        f"with a {data['composition_style']} composition. "
        f"Rendered in 768x768, cinematic lighting, textured brushwork, Van Gogh style, and borderless."
    )

def generate_image_from_prompt(prompt):
    print(f"🎨 Generating image with prompt: {prompt}")
    image = pipeline(prompt, height=768, width=768, num_inference_steps=30, guidance_scale=8.0).images[0]
    return image

def process_text_input(input_image):
    temp_path = "user_input.jpg"
    input_image.save(temp_path)
    processed_path = preprocess_handwriting(temp_path)

    extracted_text, detected_language = extract_text_from_handwriting(processed_path)
    translated_text = translate_to_english(extracted_text)
    emotion_info = analyze_emotion(translated_text)
    composition_style = assign_composition_style(emotion_info["emotion"])
    cultural_info = fetch_cultural_prompt_info(detected_language)

    extracted_data = {
        "language": detected_language,
        "extracted_text": extracted_text,
        "translated_text": translated_text,
        "emotion": emotion_info["emotion"],
        "confidence": emotion_info["confidence"],
        "composition_style": composition_style,
        "culture": cultural_info["culture"],
        "art_form": cultural_info["art_form"],
        "motif": cultural_info["motif"],
        "script": cultural_info["script"]
    }

    prompt = build_prompt(extracted_data)
    image = generate_image_from_prompt(prompt)

    return {
        "generated_image": image,
        "extracted_data": extracted_data
    }
