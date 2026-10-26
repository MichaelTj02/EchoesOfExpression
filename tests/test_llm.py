import base64
import io

import pytest
from PIL import Image

from backend import llm
from conftest import CULTURE, EMOTION, OCR, PAINTERS, TRANSLATE, VISUAL_GOAL, FakeOpenAIClient


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
    client = FakeOpenAIClient({TRANSLATE: '{"translation": "  Hello world  "}'})
    assert llm.translate_to_english(client, "Hola mundo", "Spanish") == "Hello world"
    assert "Spanish text" in client.prompts()[0]


@pytest.mark.parametrize("response", ["Hello world, no JSON here", '{"translation": ""}', '{"other": "x"}'])
def test_translate_to_english_bad_reply_returns_original(response):
    client = FakeOpenAIClient({TRANSLATE: response})
    assert llm.translate_to_english(client, "Hola mundo") == "Hola mundo"


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


# suggest_painters

def test_suggest_painters_happy_path():
    client = FakeOpenAIClient({
        PAINTERS: '```json\n{"painters": [{"name": " Claude Monet ", "style": "impressionist, soft light"}, '
                  '{"name": "Henri Matisse"}]}\n```'
    })
    assert llm.suggest_painters(client, "French", count=2) == [
        {"name": "Claude Monet", "style": "impressionist, soft light"},
        {"name": "Henri Matisse", "style": ""},
    ]
    assert any("French" in prompt and "2 different famous painters" in prompt for prompt in client.prompts())


def test_suggest_painters_drops_malformed_entries():
    client = FakeOpenAIClient({
        PAINTERS: '{"painters": ["Monet", {"style": "no name"}, {"name": ""}, {"name": "Edgar Degas", "style": null}]}'
    })
    assert llm.suggest_painters(client, "French") == [{"name": "Edgar Degas", "style": ""}]


@pytest.mark.parametrize("response", [
    RuntimeError("boom"),
    "not json at all",
    '{"painters": "Claude Monet"}',
    '{"artists": []}',
])
def test_suggest_painters_failure_returns_empty(response):
    client = FakeOpenAIClient({PAINTERS: response})
    assert llm.suggest_painters(client, "French") == []


def test_suggest_painters_unknown_language_skips_api():
    client = FakeOpenAIClient()
    assert llm.suggest_painters(client, "Unknown") == []
    assert client.calls == []


# check_connection

def test_check_connection_returns_reply():
    client = FakeOpenAIClient(default="Hi there")
    assert llm.check_connection(client) == "Hi there"
    assert len(client.calls) == 1


# make_client

@pytest.fixture
def no_llm_env(monkeypatch, tmp_path):
    pytest.importorskip("openai")
    dotenv = pytest.importorskip("dotenv")
    for name in ("OPENAI_API_KEY", "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    # make_client imports load_dotenv lazily, so this stops it finding a real .env
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *args, **kwargs: False)
    return monkeypatch


def test_make_client_defaults_to_local_server_without_key(no_llm_env):
    client = llm.make_client()
    assert str(client.base_url).startswith(llm.LLM_BASE_URL)


def test_make_client_openai_without_key_raises(no_llm_env):
    with pytest.raises(llm.MissingAPIKeyError):
        llm.make_client(base_url="https://api.openai.com/v1")


def test_make_client_reads_base_url_and_key_from_env(no_llm_env):
    no_llm_env.setenv("LLM_BASE_URL", "https://api.openai.com/v1")
    no_llm_env.setenv("OPENAI_API_KEY", "sk-test-key")
    client = llm.make_client()
    assert "api.openai.com" in str(client.base_url)
    assert client.api_key == "sk-test-key"


def test_make_client_with_explicit_key(no_llm_env):
    import openai

    client = llm.make_client("sk-test-key")
    assert isinstance(client, openai.OpenAI)


def test_llm_model_env_override(monkeypatch):
    monkeypatch.delenv("LLM_MODEL", raising=False)
    assert llm.llm_model() == llm.LLM_MODEL
    monkeypatch.setenv("LLM_MODEL", "gemma3:12b")
    assert llm.llm_model() == "gemma3:12b"


def test_check_connection_unreachable_server_gives_clear_error(no_llm_env):
    # Nothing listens on port 9, so the real client fails to connect
    client = llm.make_client(base_url="http://127.0.0.1:9/v1")
    client = client.with_options(max_retries=0, timeout=2)
    with pytest.raises(llm.LLMConnectionError, match="Could not reach"):
        llm.check_connection(client)


# normalize_language

@pytest.mark.parametrize("raw, expected", [
    ("English", "English"),
    ("en", "English"),
    ("polish", "Polish"),
    (" JA ", "Japanese"),
    ("unknown", "Unknown"),
    ("", "Unknown"),
    (None, "Unknown"),
])
def test_normalize_language(raw, expected):
    assert llm.normalize_language(raw) == expected


def test_extract_text_normalizes_language_code():
    client = FakeOpenAIClient({OCR: '{"text": "I am happy", "language": "en"}'})
    assert llm.extract_text(client, "data:image/png;base64,AAAA") == ("I am happy", "English")


# normalize_emotion

@pytest.mark.parametrize("raw, expected", [
    ("Happiness", "Happiness"),
    ("sadness", "Sadness"),
    (" ANGER ", "Anger"),
    ("joy", "Happiness"),
    ("Afraid", "Fear"),
    ("Neutral", "Unknown"),
    (None, "Unknown"),
])
def test_normalize_emotion(raw, expected):
    assert llm.normalize_emotion(raw) == expected


def test_analyze_emotion_normalizes_label_and_confidence():
    client = FakeOpenAIClient(
        {EMOTION: '{"emotion": "joy", "confidence": "0.7", "language": "English"}'}
    )
    result = llm.analyze_emotion(client, "yay")
    assert result["emotion"] == "Happiness"
    assert result["confidence"] == 0.7


def test_json_stages_request_json_mode():
    client = FakeOpenAIClient({
        OCR: '{"text": "hi", "language": "English"}',
        EMOTION: '{"emotion": "Fear", "confidence": 0.5, "language": "English"}',
        CULTURE: '{"culture": "a", "art_form": "b", "motif": "c", "script": "d"}',
        PAINTERS: '{"painters": []}',
    })
    llm.extract_text(client, "data:image/png;base64,AAAA")
    llm.analyze_emotion(client, "hi")
    llm.fetch_cultural_info(client, "English")
    llm.suggest_painters(client, "English")
    assert all(call.get("response_format") == {"type": "json_object"} for call in client.calls)


def test_missing_api_key_error_is_exported():
    import backend

    assert backend.MissingAPIKeyError is llm.MissingAPIKeyError
    assert issubclass(llm.MissingAPIKeyError, RuntimeError)
