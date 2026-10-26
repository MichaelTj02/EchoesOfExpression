# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Echoes of Expression (IAT460 course project): turns a photo of handwriting into AI-generated art. A vision LLM (default: local `qwen2.5vl:7b` via Ollama; any OpenAI-compatible server works) reads the handwriting and infers language/emotion/culture, Stable Diffusion XL renders an image from that, and a reactive "agent" gradually paints the image onto a shared 1920x1080 canvas that accumulates across uploads.

## Commands

```bash
pip install -r requirements.txt
python main.py                                            # Tkinter GUI
python -m backend handwritingImages/handwriting1.png      # headless run, optional --steps N
python -m pytest                                          # all tests (no GPU / network / API key needed)
python -m pytest tests/test_engine.py::test_process_english         # single test
```

- LLM: defaults to Ollama at `http://localhost:11434/v1` (`ollama pull qwen2.5vl:7b`). Override with `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` (or `OPENAI_API_KEY`) in `.env`; see `.env.example`. A key is only required when the base URL is api.openai.com (then the GUI asks via a dialog).
- First run downloads `stabilityai/stable-diffusion-xl-base-1.0` (fp16, ~7GB). SD 2.1 was removed from Hugging Face, which is why the project moved to SDXL. Needs a GPU with ~8GB+ VRAM; device is chosen CUDA → MPS → CPU.
- Outputs go to `image/` (created automatically): `generated_image.png`, `final_collaborative_canvas.png`.
- Sample inputs live in `handwritingImages/`; reference outputs in `sampleOutput/`.

## Architecture

Backend (`backend/`) and frontend (`gui/`) are separated; the GUI only talks to `backend.EchoesEngine`.

- `backend/engine.py`: `EchoesEngine` orchestrates everything. `load()` creates the LLM client, checks the connection, and loads SD. `process(image, on_progress, on_frame, draw_steps)` runs the stages (`STAGES`: ocr → translate (non-English only) → emotion → culture → generate → draw → done) and returns a `PipelineResult` dataclass. A lock serializes `process` calls because the canvas and output files are shared. Client, image generator and canvas are constructor-injectable (tests use fakes).
- `backend/llm.py`: stateless LLM stage functions taking `client` plus inputs, using the `openai` client against whatever server `make_client` points at. The model name comes from `llm_model()` (env override). JSON stages request `response_format=json_object`, and all JSON replies go through `parse_json`. `analyze_emotion` normalizes labels via `normalize_emotion` (case, synonyms, unknown → `Unknown`). `suggest_painters` asks for `PAINTER_CHOICES` famous painters from the language's region and the engine picks one at random per upload, so the style varies. `check_connection` turns unreachable-server / missing-model errors into `LLMConnectionError`. If OCR returns language `Unknown`, the engine falls back to the language from `analyze_emotion`.
- `backend/prompting.py`: `COMPOSITION_MAPPING`, `painting_style` and `build_prompt` (pure).
- `backend/image_gen.py`: `ImageGenerator` wraps the SDXL pipeline (`StableDiffusionXLPipeline`). Sizes come from `SD_SIZE_OPTIONS` (SDXL-native ~1MP); `AGENT_RESIZE_RATIO` is tuned to that size, so change both together.
- `backend/canvas.py`: `CollaborativeCanvas` owns the canvas image, agents, placed regions and the upload counter. `run()` drives agents and emits `on_frame(flattened copy)` every `FRAME_EVERY` steps, sleeping `frame_delay` after each frame. That delay is what paces the visible animation.
- `backend/agent.py`: `Agent` paints its source image patch-by-patch from the center outward. Emotion picks a `shape_mode` that controls opacity falloff and neighbor spreading.
- `backend/config.py`: all tunables (canvas size, SD settings, placement/overlap limits, draw steps, frame pacing, model ids, output paths).
- `gui/app.py`: `CulturalAI`. Worker threads never touch Tk; they post events to a `queue.Queue` and write the newest frame to a single slot, and `_poll` (via `root.after`) applies them on the main thread. The upload button is disabled while loading or processing.

### Invariants

- Imports must stay cheap: `torch`/`diffusers` are imported inside `ImageGenerator.load`, and `openai`/`dotenv` inside `llm.make_client`. `tests/test_imports.py` enforces this.
- Emotion labels must match across `EMOTIONS` (`config.py`), the `analyze_emotion` prompt (`llm.py`), `COMPOSITION_MAPPING` (`prompting.py`) and `EMOTION_TO_SHAPE` (`agent.py`).
- The canvas persists across uploads: each `run()` increments `input_counter`, and an agent keeps drawing for `max_rounds` (4) later runs before it is dropped. `used_regions` is never cleared, so `add_agent` can fail (returns `False`) once the canvas fills up.
- Presentation (e.g. `format_summary`) lives in `gui/`, not the backend.
