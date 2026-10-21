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


def composition_for(emotion: str) -> str:
    """Assigns a composition layout based on the detected emotion."""
    return COMPOSITION_MAPPING.get(emotion, COMPOSITION_MAPPING["Unknown"])


def build_prompt(cultural_info: dict, composition_style: str) -> str:
    culture = cultural_info["culture"]
    art_form = cultural_info["art_form"]
    motif = cultural_info["motif"]
    script = cultural_info["script"]

    prompt = (
        f"A high-resolution digital painting of a symbolic {culture} scene or object, "
        f"inspired by traditional {art_form}, "
        f"featuring {motif}, layered with {script}, "
        f"with a strong {composition_style} image composition. "
        f"Rendered in cinematic lighting, textured brushwork, Van Gogh style, and borderless."
    )

    return prompt
