from backend.prompting import DEFAULT_PAINTING_STYLE, COMPOSITION_MAPPING, build_prompt, composition_for, painting_style


def test_composition_for_known_emotion():
    assert composition_for("Happiness") == "Symmetrical & Balanced"
    assert composition_for("Anger") == "Chaotic & Overlapping"


def test_composition_for_unknown_emotion_falls_back():
    assert composition_for("Boredom") == COMPOSITION_MAPPING["Unknown"]
    assert composition_for("Unknown") == "Abstract Freeform"


def test_build_prompt_includes_all_parts():
    info = {
        "culture": "Japanese",
        "art_form": "ukiyo-e woodblock",
        "motif": "cherry blossoms",
        "script": "kanji calligraphy",
    }
    prompt = build_prompt(info, "Soft, Scattered & Faded")

    for value in info.values():
        assert value in prompt
    assert "Soft, Scattered & Faded" in prompt
    assert prompt.startswith("A high-resolution digital painting")
    assert DEFAULT_PAINTING_STYLE in prompt
    assert "Van Gogh" not in prompt


def test_build_prompt_uses_painter():
    info = {"culture": "French", "art_form": "a", "motif": "b", "script": "c"}
    painter = {"name": "Henri Matisse", "style": "fauvist, bold flat color"}
    prompt = build_prompt(info, "Symmetrical & Balanced", painter)
    assert "painted in the style of Henri Matisse, fauvist, bold flat color" in prompt


def test_painting_style_without_style_description():
    assert painting_style({"name": "Claude Monet", "style": ""}) == "painted in the style of Claude Monet"
    assert painting_style(None) == DEFAULT_PAINTING_STYLE
