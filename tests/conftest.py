from types import SimpleNamespace

import pytest
from PIL import Image

# Substrings that identify which GPT prompt is being sent
OCR = "Extract the handwritten text"
TRANSLATE = "text to English"
EMOTION = "Analyze the following text"
VISUAL_GOAL = "two or three words"
CULTURE = "cultural visual designer"


def _message_text(messages):
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.extend(item.get("text", "") for item in content if isinstance(item, dict))
    return "\n".join(parts)


def _response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


class FakeOpenAIClient:
    """Mimics client.chat.completions.create, answering by matching a substring of the prompt."""

    def __init__(self, responses=None, default="Hello!"):
        self.responses = dict(responses or {})
        self.default = default
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        text = _message_text(kwargs.get("messages", []))
        for key, value in self.responses.items():
            if key in text:
                if isinstance(value, Exception):
                    raise value
                return _response(value)
        return _response(self.default)

    def prompts(self):
        return [_message_text(call.get("messages", [])) for call in self.calls]


class StubImageGenerator:
    def __init__(self, color=(200, 40, 40), size=(200, 150)):
        self.color = color
        self.size = size
        self.load_count = 0
        self.prompts = []

    @property
    def loaded(self):
        return self.load_count > 0

    def load(self):
        self.load_count += 1

    def generate(self, prompt, size=None):
        self.prompts.append(prompt)
        return Image.new("RGB", size or self.size, self.color)


def make_fake_client(
    text="Hello world",
    language="English",
    translation="Hello world (translated)",
    emotion="Happiness",
    confidence=0.9,
    analysis_language="English",
    visual_goal="Mandala patterns",
    cultural_info=None,
):
    cultural_info = cultural_info or {
        "culture": "Balinese",
        "art_form": "batik painting",
        "motif": "lotus flowers",
        "script": "Balinese script",
    }
    culture_json = (
        "{"
        + ", ".join(f'"{key}": "{value}"' for key, value in cultural_info.items())
        + "}"
    )
    return FakeOpenAIClient({
        OCR: f'{{"text": "{text}", "language": "{language}"}}',
        TRANSLATE: f'{{"translation": "{translation}"}}',
        EMOTION: (
            f'{{"emotion": "{emotion}", "confidence": {confidence}, '
            f'"language": "{analysis_language}"}}'
        ),
        VISUAL_GOAL: visual_goal,
        CULTURE: culture_json,
    })


@pytest.fixture
def fake_client():
    return make_fake_client()


@pytest.fixture
def stub_generator():
    return StubImageGenerator()


@pytest.fixture
def handwriting_png(tmp_path):
    path = tmp_path / "handwriting.png"
    Image.new("RGB", (64, 32), (255, 255, 255)).save(path)
    return path
