# For API keys
from dotenv import load_dotenv
import os

# Open AI GPT-4
import openai

# Handwriting OCR
import cv2
import numpy as np
import base64

# For sentiment analysis
import json
import re

# For stable diffusion
import torch
from diffusers import StableDiffusionPipeline

# For canvas and drawing
from PIL import Image, ImageEnhance, ImageDraw
import random
from IPython.display import display, clear_output
import time # For live drawing
from collections import deque
import cv2


# In[11]:


load_dotenv() # load .env file to get API KEY

openai.api_key = os.getenv("OPENAI_API_KEY")

# Check if API key is loaded
if not openai.api_key:
    raise ValueError("API Key not found.")


# In[12]:


# Create client 
client = openai.OpenAI(api_key=openai.api_key)


# In[13]:


# Test API call with the updated function
response = client.chat.completions.create(
    model="gpt-4-turbo",
    messages=[{"role": "user", "content": "Hello from Jupyter Notebook!"}]
)

print("API Key Loaded Successfully!")
print("ChatGPT Response:", response.choices[0].message.content)


# ## Initialize Drawing Agents (Reactive Systems)

# In[14]:


class Agent:
    def __init__(self, image, start_x, start_y, patch_size=5):
        self.image = image.convert("RGBA")  # Ensure image has alpha
        self.patch_size = patch_size
        self.origin_x = start_x
        self.origin_y = start_y
        self.visited = set()
        self.queue = deque()

        # Drawing mode determines shape of reveal
        self.drawing_mode = random.choice([
            "center_out", 
            "top_down", 
            "left_to_right", 
            "spiral", 
            "organic_noise"
        ])

        center_x = self.origin_x + self.image.width // 2
        center_y = self.origin_y + self.image.height // 2

        if self.drawing_mode == "center_out" or self.drawing_mode == "spiral":
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

    def update(self, canvas):
        if not self.queue:
            return

        x, y = self.queue.popleft()
        key = (x, y)
        if key in self.visited:
            return

        self.visited.add(key)

        # Get image patch
        patch = self.image.crop((
            max(0, x - self.origin_x),
            max(0, y - self.origin_y),
            max(0, x - self.origin_x + self.patch_size),
            max(0, y - self.origin_y + self.patch_size)
        ))

        # Fade alpha based on distance from center
        center_x = self.origin_x + self.image.width // 2
        center_y = self.origin_y + self.image.height // 2
        dist = ((x - center_x) ** 2 + (y - center_y) ** 2) ** 0.5
        max_dist = ((self.image.width // 2) ** 2 + (self.image.height // 2) ** 2) ** 0.5
        fade = max(0.2, 1.0 - (dist / max_dist))  # Avoid full transparency

        patch = patch.copy()
        r, g, b, a = patch.split()
        a = a.point(lambda p: int(p * fade))
        patch.putalpha(a)

        canvas.paste(patch, (x, y), patch)

        # Base directions (4-neighbors)
        directions = [
            (self.patch_size, 0), (-self.patch_size, 0),
            (0, self.patch_size), (0, -self.patch_size)
        ]

        # Modify expansion pattern
        if self.drawing_mode == "spiral":
            directions = sorted(directions, key=lambda d: random.random() + 0.3 * (d[0] + d[1]))

        elif self.drawing_mode == "top_down":
            directions = sorted(directions, key=lambda d: d[1])  # prioritize y (vertical)

        elif self.drawing_mode == "left_to_right":
            directions = sorted(directions, key=lambda d: d[0])  # prioritize x (horizontal)

        elif self.drawing_mode == "organic_noise":
            random.shuffle(directions)

        # Expand to neighbors
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if (nx, ny) not in self.visited:
                if self.origin_x <= nx < self.origin_x + self.image.width - self.patch_size and \
                   self.origin_y <= ny < self.origin_y + self.image.height - self.patch_size:
                    self.queue.append((nx, ny))

# Globals
CANVAS_WIDTH = 1920
CANVAS_HEIGHT = 1080
canvas = Image.new("RGBA", (CANVAS_WIDTH, CANVAS_HEIGHT), (255, 255, 255, 0))
agents = []
extracted_data = {}
used_regions = []  # to track agent locations

# ## Process Handwritten Text (OCR, Translation, and Emotion Extraction)

# In[19]:


# Text Processing methods

# Preprocess handwriting image to make it more legible
def preprocess_handwriting(image_path):
    # Load image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply adaptive thresholding (Binarization)
    processed = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Save processed image
    processed_path = "processed_handwriting.jpg"
    cv2.imwrite(processed_path, processed)

    return processed_path

# Convert image to Base64
def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")
    
# Extract handwriting image
def extract_text_from_handwriting(image_path):
    base64_image = image_to_base64(image_path)  # Convert image to base64

    try:
        # Send request to OpenAI GPT-4o
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
            json_text = json_match.group(0)  # Extract JSON-only content
        else:
            raise ValueError("Invalid JSON format received from GPT.")

        # Parse cleaned JSON response
        text_data = json.loads(json_text)
        extracted_text = text_data.get("text", "Unknown")
        detected_language = text_data.get("language", "Unknown")

        return extracted_text, detected_language

    except Exception as e:
        print(f"OpenAI API Error: {e}")
        return None, "Unknown"
    
# Detect language & translate to English using GPT-4o
def translate_to_english(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Detect the language of this text: '{text}'. If it's not English, translate it to English. If it's already in English, return the same text."}
            ],
            max_tokens=500
        )

        translated_text = response.choices[0].message.content.strip()
        return translated_text

    except Exception as e:
        print(f"Translation Error: {e}")
        return text  # If translation fails, return original text
    
# Analyze emotion
def analyze_emotion(text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": f"Analyze the following text: '{text}'. "
                 "Return ONLY a JSON object with the following structure: "
                 "{{\"emotion\": \"Happiness\", \"confidence\": 0.92, \"language\": \"English\"}}. "
                 "The 'emotion' should be one of: Happiness, Sadness, Fear, Anger, Surprise, or Disgust. "
                 "The 'confidence' should be a float between 0 and 1. "
                 "The 'language' should be a single-word language name (e.g., English, Japanese, Chinese). "
                 "No explanations, no extra text—return only a valid JSON output."}
            ],
            max_tokens=100
        )

        raw_response = response.choices[0].message.content.strip()

        # Extract JSON using regex if there's extra text
        json_match = re.search(r"\{.*\}", raw_response, re.DOTALL)
        if json_match:
            json_text = json_match.group(0)  # Extracts the JSON part only
        else:
            raise ValueError("Invalid JSON format received from GPT.")

        # Parse JSON safely
        text_analysis_data = json.loads(json_text)

        # Extract emotion, confidence, and language
        detected_emotion = text_analysis_data.get("emotion", "Unknown")
        confidence_score = text_analysis_data.get("confidence", 0.0)
        detected_language = text_analysis_data.get("language", "Unknown")

        return {
            "emotion": detected_emotion,
            "confidence": confidence_score,
            "language": detected_language
        }

    except Exception as e:
        print(f"Text Analysis Error: {e}")
        return {
            "emotion": "Unknown",
            "confidence": 0.0,
            "language": "Unknown"
        }
        
def process_handwritten_image(processed_image):
    global extracted_data  # Declare global variable
    
    extracted_text, detected_language = extract_text_from_handwriting(processed_image)

    if extracted_text:
        translated_text = translate_to_english(extracted_text) if detected_language.lower() != "english" else extracted_text
        text_analysis_data = analyze_emotion(translated_text)

        detected_emotion = text_analysis_data["emotion"]
        confidence_score = text_analysis_data["confidence"]

        # Store data in the global variable
        extracted_data = {
            "extracted_text": extracted_text,
            "language": detected_language,
            "translated_text": translated_text,
            "emotion": detected_emotion,
            "confidence": confidence_score
        }

        # Print output
        print("\n Extracted Information Stored:")
        print(f"Extracted Text: {extracted_data['extracted_text']}")
        print(f"Detected Language: {extracted_data['language']}")
        print(f"Translated Text: {extracted_data['translated_text'] if extracted_data['language'].lower() != 'english' else 'N/A'}")
        print(f"Detected Emotion: {extracted_data['emotion']}")
        print(f"Confidence Score: {extracted_data['confidence']:.2f}")

    else:
        print("\n Failed to process the text.")


# ## Map Emotion and Incorporate Culture from Language

# In[ ]:


# Visual Composition mapping based on detected emotion, grammar system
COMPOSITION_MAPPING = {
    "Happiness": "Symmetrical & Balanced",
    "Sadness": "Soft, Scattered & Faded",
    "Anger": "Chaotic & Overlapping",
    "Fear": "Dark & Enclosed",
    "Surprise": "Expanding & Explosive",
    "Disgust": "Distorted & Melting",
    "Unknown": "Abstract Freeform"
}

def assign_composition_style():
    """Assigns a composition layout based on the detected emotion."""
    global extracted_data

    detected_emotion = extracted_data.get("emotion", "Unknown")
    composition_style = COMPOSITION_MAPPING.get(detected_emotion, COMPOSITION_MAPPING["Unknown"])

    extracted_data["composition_style"] = composition_style

    return composition_style

# Determine visual goal depending on the language detected
def get_visual_goal():
    """Uses GPT-4 to generate cultural visual goal based on detected language."""
    
    global extracted_data
    
    detected_language = extracted_data.get("language", "Unknown")
    
    if detected_language == "Unknown":
        print("Language not detected")
        return "Unknown cultural element"
    
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in global art and culture."},
                {"role": "user", "content": f"Provide only **two or three words** that represent a traditional artistic style, symbol, or pattern from the culture associated with the {detected_language} language. Do not give explanations. Just return the keywords."}
            ],
            max_tokens=100
        )

        visual_goal = response.choices[0].message.content.strip()
        
        return visual_goal

    except Exception as e:
        print(f"GPT API Error: {e}")
        return "Unknown cultural element"
    
def fetch_cultural_prompt_info():
    global extracted_data

    detected_language = extracted_data.get("language", "Unknown")

    if detected_language == "Unknown":
        print("Language not detected.")
        return {
            "culture": detected_language,
            "art_form": "traditional art",
            "motif": "cultural motif",
            "script": f"{detected_language} script"
        }

    system_prompt = (
        "You are a cultural visual designer. Given a language, respond ONLY in raw JSON format with these 4 keys:\n"
        "'culture', 'art_form', 'motif', 'script'.\n"
        "Keep each value short (2–6 words). DO NOT return markdown or explanations."
    )

    user_prompt = f"The language is: {detected_language}."

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150
        )

        content = response.choices[0].message.content.strip()

        # Strip markdown/code block if GPT wraps it
        if content.startswith("```"):
            content = content.strip("`").strip()
            # If there's still a language label, remove it
            if content.startswith("json"):
                content = content[4:].strip()

        # Try parsing JSON
        return json.loads(content)

    except json.JSONDecodeError as je:
        print("⚠️ Failed to parse GPT response as JSON.")
        print("Raw response:\n", content)
        return {
            "culture": detected_language,
            "art_form": "traditional art",
            "motif": "cultural motif",
            "script": f"{detected_language} script"
        }

    except Exception as e:
        print(f"GPT API Error: {e}")
        return {
            "culture": detected_language,
            "art_form": "traditional art",
            "motif": "cultural motif",
            "script": f"{detected_language} script"
        }


# ## Stable Diffusion and Canvas for Drawing Space

# In[54]:


# Stable Diffusion 2.1
MODEL_ID = "stabilityai/stable-diffusion-2-1"

pipeline = StableDiffusionPipeline.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    variant="fp16"  # use float16 weights for efficiency
)

# Move to Apple Silicon GPU
device = "mps" if torch.backends.mps.is_available() else "cpu"
pipeline.to(device)
# Turn off NSFW filters due to overlapping stuff that may be detected as NSFW
pipeline.safety_checker = None

# Optimizations
pipeline.enable_attention_slicing()
pipeline.enable_vae_slicing()


# In[181]:

def build_dynamic_prompt():
    global extracted_data

    composition_style = extracted_data.get("composition_style", "balanced composition")
    cultural_info = fetch_cultural_prompt_info()

    culture = cultural_info["culture"]
    art_form = cultural_info["art_form"]
    motif = cultural_info["motif"]
    script = cultural_info["script"]

    prompt = (
        f"A high-resolution digital painting of a symbolic {culture} scene or object, "
        f"inspired by traditional {art_form}, "
        f"featuring {motif}, layered with {script}, "
        f"with a {composition_style} composition. "
        f"Rendered in 768x768, cinematic lighting, textured brushwork, Van Gogh style, and borderless."
    )

    return prompt

def generate_image():
    """Generates an image and assigns it to an agent for gradual drawing."""
    prompt = build_dynamic_prompt()
    
    size_options = [
        (768, 512), (512, 768), (640, 640), (768, 576),
        (576, 768), (720, 540), (540, 720), (640, 480)
    ]
    width, height = random.choice(size_options)

    # Generate the image using Stable Diffusion 2.1 
    image = pipeline(
        prompt,
        height=height,
        width=width,
        num_inference_steps=30, # slightly higher for better quality
        guidance_scale=8.0
    ).images[0]

    # Save image
    image_path = "generated_image.png"
    image.save(image_path)

    return image_path, prompt

def intersect(box1, box2):
    # function to check if agent location is overlapping
    return not (
        box1[2] <= box2[0] or  # right <= left
        box1[0] >= box2[2] or  # left >= right
        box1[3] <= box2[1] or  # bottom <= top
        box1[1] >= box2[3]     # top >= bottom
    )

def add_agent_for_image(image_path):
    global agents, canvas, used_regions

    image = Image.open(image_path).convert("RGBA")
    resize_ratio = 0.4  # Scale down to 40% of original image
    new_width = int(image.width * resize_ratio)
    new_height = int(image.height * resize_ratio)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    max_attempts = 100
    for _ in range(max_attempts):
        x = random.randint(0, canvas.width - new_width)
        y = random.randint(0, canvas.height - new_height)

        # Check overlap with other agents
        new_box = (x, y, x + 600, y + 600)
        overlap = any(intersect(new_box, region) for region in used_regions)

        if not overlap:
            used_regions.append(new_box)
            new_agent = Agent(image, x, y)
            agents.append(new_agent)
            return

    print("Could not place agent without overlap.")

def show_live_canvas(canvas):
    return True  # placeholder, no-op now

def run_live_drawing_loop(steps=5000, delay=0.01, update_callback=None):
    for step in range(steps):
        for agent in agents:
            agent.update(canvas)

        if update_callback and step % 5 == 0:
            update_callback(canvas)  # Push update to GUI

        time.sleep(delay)

    canvas.save("final_collaborative_canvas.png")
    print("Canvas saved as final_collaborative_canvas.png")

def automate_from_image_file(image_input, update_callback=None):
    global extracted_data, canvas, agents
    
    extracted_data = {} # clear old values

    # Determine if it's a file path (str) or an Image object
    if isinstance(image_input, str):
        image_path = image_input
    else:
        image_path = "uploaded_image.jpg"
        image_input.save(image_path)

    # Extract handwritten text & language
    extracted_text, detected_language = extract_text_from_handwriting(image_path)
    if not extracted_text:
        return "Could not extract any text from image", None

    translated_text = translate_to_english(extracted_text) if detected_language.lower() != "english" else extracted_text
    text_analysis_data = analyze_emotion(translated_text)
    composition_style = assign_composition_style()
    visual_goal = get_visual_goal()

    extracted_data = {
        "extracted_text": extracted_text,
        "language": detected_language,
        "translated_text": translated_text,
        "emotion": text_analysis_data["emotion"],
        "confidence": text_analysis_data["confidence"],
        "composition_style": composition_style,
        "visual_goal": visual_goal
    }

    # Generate and draw
    generated_image_path, prompt = generate_image()
    agents.clear()
    add_agent_for_image(generated_image_path)
    run_live_drawing_loop(update_callback=update_callback)

    summary = (
        f"Text: {extracted_text}\n"
        f"Language: {detected_language}\n"
        f"Translated: {translated_text if detected_language.lower() != 'english' else 'Already in English'}\n"
        f"Emotion: {text_analysis_data['emotion']} (Confidence: {text_analysis_data['confidence']:.2f})\n"
        f"Composition: {composition_style}\n"
        f"Cultural Visual: {visual_goal}\n"
        f"Prompt: {prompt}"
    )

    return summary, canvas
