from __future__ import annotations

import base64
import io
import json
import logging
import mimetypes
import os
import re
from pathlib import Path

from PIL import Image

from .config import EMOTIONS, LLM_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)


class MissingAPIKeyError(RuntimeError):
    pass


class LLMConnectionError(RuntimeError):
    pass


# Model can be overridden in .env; make_client loads .env before any LLM call
def llm_model() -> str:
    return os.getenv("LLM_MODEL") or LLM_MODEL


def make_client(api_key: str | None = None, base_url: str | None = None) -> "openai.OpenAI":
    # Imported here so the rest of the backend can be used without these installed
    import openai
    from dotenv import load_dotenv

    load_dotenv()  # load .env file for LLM_* / OPENAI_API_KEY settings
    base_url = base_url or os.getenv("LLM_BASE_URL") or LLM_BASE_URL
    api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY")

    if not api_key:
        if "api.openai.com" in base_url:
            raise MissingAPIKeyError("OpenAI API key not found. Set OPENAI_API_KEY in a .env file.")
        api_key = "local"  # local servers like Ollama ignore the key, but the client requires one

    logger.info("Using LLM %s at %s", llm_model(), base_url)
    return openai.OpenAI(api_key=api_key, base_url=base_url)


def check_connection(client) -> str:
    try:
        response = client.chat.completions.create(
            model=llm_model(),
            messages=[{"role": "user", "content": "Hello!"}],
            max_tokens=20
        )
    except Exception as e:
        import openai  # only reached when the real client raised

        # Turn the common local-server failures into actionable messages
        if isinstance(e, openai.APIConnectionError):
            raise LLMConnectionError(
                f"Could not reach the LLM server at {client.base_url}. "
                "If you're using Ollama, make sure it is running."
            ) from e
        if isinstance(e, openai.NotFoundError):
            raise LLMConnectionError(
                f"Model '{llm_model()}' not found at {client.base_url}. "
                f"For Ollama, run: ollama pull {llm_model()}"
            ) from e
        raise

    return response.choices[0].message.content


# Small models don't always stick to the exact labels
EMOTION_SYNONYMS = {
    "joy": "Happiness", "happy": "Happiness",
    "sad": "Sadness", "sorrow": "Sadness",
    "afraid": "Fear", "fearful": "Fear", "anxiety": "Fear",
    "angry": "Anger", "rage": "Anger",
    "surprised": "Surprise",
    "disgusted": "Disgust",
}


# Small models sometimes answer with ISO codes or lowercase names
LANGUAGE_CODES = {
    "en": "English", "es": "Spanish", "fr": "French", "de": "German", "it": "Italian",
    "pt": "Portuguese", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ar": "Arabic", "hi": "Hindi", "id": "Indonesian", "th": "Thai", "pl": "Polish",
    "tr": "Turkish", "vi": "Vietnamese", "nl": "Dutch", "ky": "Kyrgyz",
}


def normalize_language(value) -> str:
    language = str(value or "").strip()
    if not language or language.lower() == "unknown":
        return "Unknown"
    return LANGUAGE_CODES.get(language.lower(), language.title())


def normalize_emotion(value) -> str:
    emotion = str(value).strip().capitalize()
    if emotion in EMOTIONS:
        return emotion
    return EMOTION_SYNONYMS.get(emotion.lower(), "Unknown")


def parse_json(raw: str) -> dict:
    content = raw.strip()

    # Strip markdown/code block if GPT wraps it
    if content.startswith("```"):
        content = content.strip("`").strip()
        # If there's still a language label, remove it
        if content.startswith("json"):
            content = content[4:].strip()

    # Extract JSON using regex if there's extra text
    json_match = re.search(r"\{.*\}", content, re.DOTALL)
    if not json_match:
        raise ValueError("Invalid JSON format received from GPT.")

    return json.loads(json_match.group(0))


# Convert image to a base64 data URL
def image_to_data_url(image) -> str:
    if isinstance(image, Image.Image):
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        data, mime_type = buffer.getvalue(), "image/png"
    else:
        data = Path(image).read_bytes()
        mime_type = mimetypes.guess_type(str(image))[0] or "image/jpeg"

    return f"data:{mime_type};base64,{base64.b64encode(data).decode('utf-8')}"


# Extract handwriting image
def extract_text(client, data_url: str) -> tuple[str | None, str]:
    try:
        response = client.chat.completions.create(
            model=llm_model(),
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text":
                            "Extract the handwritten text from this image and detect its language. "
                            "Return the result in JSON format: {\"text\": \"extracted text\", \"language\": \"detected language\"}. "
                            "The language must be its full English name (e.g. English, Spanish, Japanese), not a code. "
                            "Do not add extra explanations, just return valid JSON."
                        },
                        {"type": "image_url", "image_url": {"url": data_url}}
                    ]
                }
            ],
            max_tokens=500,
            temperature=0,
            response_format={"type": "json_object"}
        )

        raw_response = response.choices[0].message.content.strip()
        text_data = parse_json(raw_response)

        extracted_text = text_data.get("text", "Unknown")
        detected_language = normalize_language(text_data.get("language", "Unknown"))
        if detected_language == "Unknown":
            logger.info("OCR did not detect a language. Raw response: %s", raw_response)

        return extracted_text, detected_language

    except Exception as e:
        logger.error("OCR Error: %s", e)
        return None, "Unknown"


# Translate to English
def translate_to_english(client, text: str, language: str = "Unknown") -> str:
    source = f"{language} " if language != "Unknown" else ""
    try:
        response = client.chat.completions.create(
            model=llm_model(),
            messages=[
                {"role": "user", "content": f"Translate this {source}text to English: '{text}'. "
                 "Return ONLY a JSON object: {\"translation\": \"the English translation\"}. No explanations."}
            ],
            max_tokens=500,
            temperature=0,
            response_format={"type": "json_object"}
        )

        translation = parse_json(response.choices[0].message.content).get("translation")
        if not isinstance(translation, str) or not translation.strip():
            raise ValueError("No translation in response.")
        return translation.strip()

    except Exception as e:
        logger.error("Translation Error: %s", e)
        return text  # If translation fails, return original text


# Analyze emotion
def analyze_emotion(client, text: str) -> dict:
    try:
        response = client.chat.completions.create(
            model=llm_model(),
            messages=[
                {"role": "user", "content": f"Analyze the following text: '{text}'. "
                 "Return ONLY a JSON object with the following structure: "
                 "{\"emotion\": \"<emotion>\", \"confidence\": <number>, \"language\": \"<language>\"}. "
                 "The 'emotion' should be one of: Happiness, Sadness, Fear, Anger, Surprise, or Disgust. "
                 "The 'confidence' should be a float between 0 and 1. "
                 "The 'language' should be a single-word language name (e.g., English, Japanese, Chinese). "
                 "No explanations, no extra text—return only a valid JSON output."}
            ],
            max_tokens=100,
            temperature=0,
            response_format={"type": "json_object"}
        )

        text_analysis_data = parse_json(response.choices[0].message.content)

        try:
            confidence = float(text_analysis_data.get("confidence", 0.0))
        except (TypeError, ValueError):
            confidence = 0.0

        return {
            "emotion": normalize_emotion(text_analysis_data.get("emotion", "Unknown")),
            "confidence": confidence,
            "language": normalize_language(text_analysis_data.get("language", "Unknown"))
        }

    except Exception as e:
        logger.error("Text Analysis Error: %s", e)
        return {
            "emotion": "Unknown",
            "confidence": 0.0,
            "language": "Unknown"
        }


# Determine visual goal depending on the language detected
def get_visual_goal(client, language: str) -> str:
    """Uses GPT-4 to generate cultural visual goal based on detected language."""
    if language == "Unknown":
        logger.info("Language not detected")
        return "Unknown cultural element"

    try:
        response = client.chat.completions.create(
            model=llm_model(),
            messages=[
                {"role": "system", "content": "You are an expert in global art and culture."},
                {"role": "user", "content": f"Provide only **two or three words** that represent a traditional artistic style, symbol, or pattern from the culture associated with the {language} language. Do not give explanations. Just return the keywords."}
            ],
            max_tokens=100
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.error("GPT API Error: %s", e)
        return "Unknown cultural element"


def _default_cultural_info(language: str) -> dict:
    return {
        "culture": language,
        "art_form": "traditional art",
        "motif": "cultural motif",
        "script": f"{language} script"
    }


def fetch_cultural_info(client, language: str) -> dict:
    if language == "Unknown":
        logger.info("Language not detected.")
        return _default_cultural_info(language)

    system_prompt = (
        "You are a cultural visual designer. Given a language, respond ONLY in raw JSON format with these 4 keys:\n"
        "'culture', 'art_form', 'motif', 'script'.\n"
        "Keep each value short (2–6 words). DO NOT return markdown or explanations."
    )

    user_prompt = f"The language is: {language}."

    content = ""
    try:
        response = client.chat.completions.create(
            model=llm_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            max_tokens=150,
            response_format={"type": "json_object"}
        )

        content = response.choices[0].message.content
        # Fill any keys GPT left out so build_prompt always has all four
        return {**_default_cultural_info(language), **parse_json(content)}

    except ValueError:
        logger.error("Failed to parse GPT response as JSON. Raw response:\n%s", content)
        return _default_cultural_info(language)

    except Exception as e:
        logger.error("GPT API Error: %s", e)
        return _default_cultural_info(language)
