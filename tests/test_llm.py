import base64
import io

import pytest
from PIL import Image

from backend import llm
from conftest import CULTURE, EMOTION, OCR, TRANSLATE, VISUAL_GOAL, FakeOpenAIClient


# parse_json

def test_parse_json_plain():
    assert llm.parse_json('{"a": 1, "b": "two"}') == {"a": 1, "b": "two"}


def test_parse_json_fenced():
    assert llm.parse_json('```\n{"a": 1}\n```') == {"a": 1}


def test_parse_json_json_fenced():
    assert llm.parse_json('```json\n{"emotion": "Fear"}\n```') == {"emotion": "Fear"}


def test_parse_json_surrounded_by_prose():
    raw = 'Sure! Here is the result:\n{"text": "hi", "language": "English"}\nHope that helps.'
    assert llm.parse_json(raw) == {"text": "hi", "language": "English"}


@pytest.mark.parametrize("raw", ["no json here", "{not: valid json}", ""])
def test_parse_json_invalid_raises(raw):
    with pytest.raises(ValueError):
        llm.parse_json(raw)


# image_to_data_url

def _decode(data_url):
    header, payload = data_url.split(",", 1)
    return header, base64.b64decode(payload)


def test_image_to_data_url_png_path(handwriting_png):
    header, payload = _decode(llm.image_to_data_url(handwriting_png))
    assert header == "data:image/png;base64"
    assert payload == handwriting_png.read_bytes()


def test_image_to_data_url_accepts_str_path(handwriting_png):
    assert llm.image_to_data_url(str(handwriting_png)) == llm.image_to_data_url(handwriting_png)


def test_image_to_data_url_jpeg_path(tmp_path):
    path = tmp_path / "note.jpg"
    Image.new("RGB", (10, 10), (0, 0, 0)).save(path)
    header, _ = _decode(llm.image_to_data_url(path))
    assert header == "data:image/jpeg;base64"


def test_image_to_data_url_pil_image():
    header, payload = _decode(llm.image_to_data_url(Image.new("RGB", (12, 8), (1, 2, 3))))
    assert header == "data:image/png;base64"
    with Image.open(io.BytesIO(payload)) as decoded:
        assert decoded.format == "PNG"
        assert decoded.size == (12, 8)


# extract_text

def test_extract_text_happy_path():
    client = FakeOpenAIClient({OCR: '```json\n{"text": "hola mundo", "language": "Spanish"}\n```'})
    assert llm.extract_text(client, "data:image/png;base64,AAAA") == ("hola mundo", "Spanish")

    content = client.calls[0]["messages"][0]["content"]
    image_parts = [part for part in content if part.get("type") == "image_url"]
    assert image_parts[0]["image_url"]["url"] == "data:image/png;base64,AAAA"


def test_extract_text_api_error_falls_back():
    client = FakeOpenAIClient({OCR: RuntimeError("boom")})
    assert llm.extract_text(client, "data:image/png;base64,AAAA") == (None, "Unknown")


def test_extract_text_invalid_json_falls_back():
    client = FakeOpenAIClient({OCR: "I can't read that, sorry."})
    assert llm.extract_text(client, "data:image/png;base64,AAAA") == (None, "Unknown")


# translate_to_english

def test_translate_to_english_happy_path():
    client = FakeOpenAIClient({TRANSLATE: "  Hello world  "})
    assert llm.translate_to_english(client, "Hola mundo") == "Hello world"


def test_translate_to_english_failure_returns_original():
    client = FakeOpenAIClient({TRANSLATE: RuntimeError("boom")})
    assert llm.translate_to_english(client, "Hola mundo") == "Hola mundo"


# analyze_emotion

def test_analyze_emotion_happy_path():
    client = FakeOpenAIClient(
        {EMOTION: 'Result: {"emotion": "Sadness", "confidence": 0.81, "language": "French"}'}
    )
    assert llm.analyze_emotion(client, "je suis triste") == {
        "emotion": "Sadness",
        "confidence": 0.81,
        "language": "French",
    }


@pytest.mark.parametrize("response", [RuntimeError("boom"), "not json"])
def test_analyze_emotion_failure_falls_back(response):
    client = FakeOpenAIClient({EMOTION: response})
    assert llm.analyze_emotion(client, "text") == {
        "emotion": "Unknown",
        "confidence": 0.0,
        "language": "Unknown",
    }


# get_visual_goal

def test_get_visual_goal_happy_path():
    client = FakeOpenAIClient({VISUAL_GOAL: " Celtic knotwork \n"})
    assert llm.get_visual_goal(client, "Irish") == "Celtic knotwork"


def test_get_visual_goal_unknown_language_skips_api():
    client = FakeOpenAIClient()
    assert llm.get_visual_goal(client, "Unknown") == "Unknown cultural element"
    assert client.calls == []


def test_get_visual_goal_failure_falls_back():
    client = FakeOpenAIClient({VISUAL_GOAL: RuntimeError("boom")})
    assert llm.get_visual_goal(client, "Irish") == "Unknown cultural element"


# fetch_cultural_info

CULTURE_KEYS = {"culture", "art_form", "motif", "script"}


def test_fetch_cultural_info_happy_path():
    client = FakeOpenAIClient({
        CULTURE: '```json\n{"culture": "Korean", "art_form": "minhwa folk painting", '
                 '"motif": "tigers and magpies", "script": "Hangul"}\n```'
    })
    info = llm.fetch_cultural_info(client, "Korean")
    assert info == {
        "culture": "Korean",
        "art_form": "minhwa folk painting",
        "motif": "tigers and magpies",
        "script": "Hangul",
    }
    assert any("Korean" in prompt for prompt in client.prompts())


def _fallback(language):
    return {
        "culture": language,
        "art_form": "traditional art",
        "motif": "cultural motif",
        "script": f"{language} script",
    }


@pytest.mark.parametrize("response", [RuntimeError("boom"), "not json at all"])
def test_fetch_cultural_info_failure_falls_back(response):
    client = FakeOpenAIClient({CULTURE: response})
    assert llm.fetch_cultural_info(client, "Korean") == _fallback("Korean")


def test_fetch_cultural_info_unknown_language_skips_api():
    client = FakeOpenAIClient()
    assert llm.fetch_cultural_info(client, "Unknown") == _fallback("Unknown")
    assert client.calls == []


# check_connection

def test_check_connection_returns_reply():
    client = FakeOpenAIClient(default="Hi there")
    assert llm.check_connection(client) == "Hi there"
    assert len(client.calls) == 1


# make_client

def test_make_client_without_key_raises(monkeypatch, tmp_path):
    pytest.importorskip("openai")
    dotenv = pytest.importorskip("dotenv")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)
    # make_client imports load_dotenv lazily, so this stops it finding a real .env
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)

    with pytest.raises(llm.MissingAPIKeyError):
        llm.make_client()


def test_make_client_with_explicit_key(monkeypatch):
    openai = pytest.importorskip("openai")
    pytest.importorskip("dotenv")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = llm.make_client("sk-test-key")
    assert isinstance(client, openai.OpenAI)


def test_missing_api_key_error_is_exported():
    import backend

    assert backend.MissingAPIKeyError is llm.MissingAPIKeyError
    assert issubclass(llm.MissingAPIKeyError, RuntimeError)
