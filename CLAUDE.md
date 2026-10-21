# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Echoes of Expression (IAT460 course project): a Tkinter app that turns a photo of handwriting into AI-generated art. GPT-4o reads the handwriting and infers language/emotion/culture, Stable Diffusion 2.1 renders an image from that, and a reactive "agent" gradually paints the image onto a shared 1920x1080 canvas that accumulates across uploads.

## Running

```bash
pip install -r requirements.txt
mkdir image      
python main.py
```

- OpenAI key: put `OPENAI_API_KEY=...` in a `.env` at the repo root. If no `.env` file is found, `OpenAIAgent` falls back to `input()` on the terminal (the GUI does not prompt).
- First run downloads `stabilityai/stable-diffusion-2-1` (fp16) via `diffusers`. Needs a GPU with ~6GB+ VRAM; device is chosen CUDA → MPS → CPU.
- Sample inputs live in `handwritingImages/`; reference outputs in `sampleOutput/`.
- There are no tests, linter, or build step.

## Architecture

Four modules, flat layout:

- `main.py` — Tkinter GUI (`CulturalAI`). The upload button spawns a daemon thread that calls `project.automate_from_image_file(path, update_callback=...)`. The callback receives the canvas, writes it to `image/latest_canvas_preview.png`, reloads it, and updates the preview label.
- `project.py` — the whole pipeline plus all shared state. `automate_from_image_file` runs, in order: GPT-4o OCR + language detection → translate to English (if needed) → emotion classification → `COMPOSITION_MAPPING` lookup → cultural visual goal / cultural prompt info (more GPT-4o calls) → `build_dynamic_prompt` → SD image (random size from `size_options`, saved to `image/generated_image.png`) → `add_agent_for_image` → `run_live_drawing_loop`. Returns `(summary_text, canvas)`.
- `agent.py` — `Agent` paints its source image onto the canvas patch-by-patch via a BFS-style queue starting at the image center. The detected emotion picks a `shape_mode` (`EMOTION_TO_SHAPE`), which controls both the per-patch opacity falloff (`SHAPE_PATTERNS`) and how neighbor points are queued.
- `openAIAgent.py` — loads the API key and builds the `openai.OpenAI` client used by `project.py`.

Emotion is the main control signal and is keyed by the same label set in three places: the prompt in `analyze_emotion`, `COMPOSITION_MAPPING` (`project.py`), and `EMOTION_TO_SHAPE` (`agent.py`). Adding/renaming an emotion means updating all three.

### Shared state and lifecycle (important for refactoring)

- **Importing `project.py` has heavy side effects**: it constructs `OpenAIAgent` (may block on `input()`), and loads the SD pipeline onto the GPU at module level. Anything that imports `project` pays this cost.
- Pipeline state is module-level globals in `project.py`: `canvas`, `agents`, `used_regions`, `input_counter`, `extracted_data`. Several stage functions communicate by reading/writing `extracted_data` rather than via arguments/returns (e.g. `assign_composition_style`, `get_visual_goal`, `fetch_cultural_prompt_info` read `extracted_data["language"/"emotion"]`). Stage order therefore matters.
- The canvas is persistent across uploads (the "collaborative" part). Each call to `run_live_drawing_loop` increments `input_counter`; an agent keeps drawing for `max_rounds` (4) subsequent uploads, then is dropped. `used_regions` is never cleared, so placement gets harder over time (`add_agent_for_image` gives up after 500 attempts).
- `main.py` imports `canvas`, `CANVAS_WIDTH`, `CANVAS_HEIGHT` from `project` directly, and its callback mutates Tkinter widgets from the worker thread.
- Files written to the working directory / `image/`: `generated_image.png`, `latest_canvas_preview.png`, `final_collaborative_canvas.png`, and `uploaded_image.jpg` (when a PIL image rather than a path is passed). Only one upload should be processed at a time — concurrent runs share all of the above.

### Known quirks

- `openai_agent.test_api` in `project.py` is referenced without calling it, so the startup "ChatGPT Response" print shows a bound method, not an API response.
- `preprocess_handwriting` (OpenCV binarization) is defined but not used in the pipeline.
- `Agent.update` computes `fade` twice; the second, unguarded computation is the one that takes effect.
- The README notes "Language not detected" appearing in the terminal as a known, unresolved issue.
