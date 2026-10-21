from backend.prompting import COMPOSITION_MAPPING, build_prompt, composition_for


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
