# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Echoes of Expression (IAT460 course project): turns a photo of handwriting into AI-generated art. A vision LLM (default: local `qwen2.5vl:7b` via Ollama; any OpenAI-compatible server works) reads the handwriting and infers language/emotion/culture, Stable Diffusion 1.5 (DreamShaper 8) renders an image from that, and a reactive "agent" gradually paints the image onto a shared 1920x1080 canvas that accumulates across uploads.

## Commands

```bash
pip install -r requirements.txt
python main.py                                            # Tkinter GUI
python -m backend handwritingImages/handwriting1.png      # headless run, takes several images, optional --steps N
python -m pytest                                          # all tests (no GPU / network / API key needed)
python -m pytest tests/test_engine.py::test_process_english         # single test
```

- LLM: defaults to Ollama at `http://localhost:11434/v1` (`ollama pull qwen2.5vl:7b`). Override with `LLM_BASE_URL` / `LLM_MODEL` / `LLM_API_KEY` (or `OPENAI_API_KEY`) in `.env`; see `.env.example`. A key is only required when the base URL is api.openai.com (then the GUI asks via a dialog).
- First run downloads `Lykon/dreamshaper-8` (SD 1.5 fine-tune, fp16, ~2GB, 512x512). SDXL was dropped as too heavy; SD 2.1 was removed from Hugging Face. Uses ~3.5GB VRAM for a batch of 4; device is chosen CUDA → MPS → CPU.
- Outputs go to `image/` (created automatically): `generated_image_{i}.png` (i = position in the batch, 1-based), `final_collaborative_canvas.png`.
- Sample inputs live in `handwritingImages/`; reference outputs in `sampleOutput/`.

## Architecture

Backend (`backend/`) and frontend (`gui/`) are separated; the GUI only talks to `backend.EchoesEngine`.

- `backend/engine.py`: `EchoesEngine` orchestrates everything. `load()` creates the LLM client, checks the connection, and loads SD. `process_many(images, on_progress, on_frame, draw_steps)` runs the stages (`STAGES`: ocr → translate (non-English only) → emotion → culture → generate → draw → done) for up to `MAX_UPLOADS_PER_BATCH` images and returns one `PipelineResult` or exception per image, in input order. The LLM stages (`_analyze`) run concurrently in a `ThreadPoolExecutor(LLM_WORKERS)`; a per-image failure fills that slot and the rest continue. Then one `generate_batch` call, one `add_agent` per image and a single `canvas.run()`, so a batch's paintings are drawn simultaneously. Painters are picked after analysis in input order so a seeded `rng` stays deterministic. `process(image)` wraps `process_many([image])` and re-raises. A lock serializes batches because the canvas and output files are shared; `on_progress` may be called from worker threads. Client, image generator and canvas are constructor-injectable (tests use fakes).
- `backend/llm.py`: stateless LLM stage functions taking `client` plus inputs, using the `openai` client against whatever server `make_client` points at. The model name comes from `llm_model()` (env override). JSON stages request `response_format=json_object`, and all JSON replies go through `parse_json`. `analyze_emotion` normalizes labels via `normalize_emotion` (case, synonyms, unknown → `Unknown`). `suggest_painters` asks for `PAINTER_CHOICES` famous painters from the language's region and the engine picks one at random per upload, so the style varies. `check_connection` turns unreachable-server / missing-model errors into `LLMConnectionError`. If OCR returns language `Unknown`, the engine falls back to the language from `analyze_emotion`.
- `backend/prompting.py`: `COMPOSITION_MAPPING`, `painting_style` and `build_prompt` (pure).
- `backend/image_gen.py`: `ImageGenerator` wraps `AutoPipelineForText2Image`, so swapping `SD_MODEL_ID` is enough to change models. `generate_batch(prompts)` runs `SD_BATCH_SIZE` prompts per GPU pass with `SD_NEGATIVE_PROMPT`, and regenerates images the safety checker blacked out (`SD_NSFW_RETRIES`; DreamShaper trips it fairly often). Size is `SD_SIZE` (512x512); `AGENT_RESIZE_RATIO` is tuned to that size, so change both together.
- `backend/canvas.py`: `CollaborativeCanvas` owns the canvas image, agents, placed regions and the upload counter. `run(steps=None)` drives all agents until each one is `done` (capped at `MAX_DRAW_STEPS`; `steps=N` caps it lower and unfinished agents continue next run), removes finished agents, emits `on_frame(flattened copy)` every `FRAME_EVERY` steps (sleeping `frame_delay` after each) plus a final frame, and returns the steps taken. That delay is what paces the visible animation; `PATCHES_PER_UPDATE` sets how much each agent paints per step (~4000 steps for a 410px painting).
- `backend/agent.py`: `Agent` paints its source image patch-by-patch from the center outward. Emotion picks a `shape_mode` that controls opacity falloff and neighbor spreading.
- `backend/config.py`: all tunables (canvas size, SD settings, placement/overlap limits, draw steps, frame pacing, model ids, output paths).
- `gui/app.py`: `CulturalAI`. Worker threads never touch Tk; they post events to a `queue.Queue` and write the newest frame to a single slot, and `_poll` (via `root.after`) applies them on the main thread. The upload button opens a multi-select dialog (capped at `MAX_UPLOADS_PER_BATCH`) and is disabled while loading or processing; per-image failures are logged, only batch-level errors show a dialog.

### Invariants

- Imports must stay cheap: `torch`/`diffusers` are imported inside `ImageGenerator.load`, and `openai`/`dotenv` inside `llm.make_client`. `tests/test_imports.py` enforces this.
- Emotion labels must match across `EMOTIONS` (`config.py`), the `analyze_emotion` prompt (`llm.py`), `COMPOSITION_MAPPING` (`prompting.py`) and `EMOTION_TO_SHAPE` (`agent.py`).
- The canvas persists across uploads: each `run()` increments `input_counter`, and agents finish their painting within the run they were added in. `used_regions` is never cleared, so `add_agent` can fail (returns `False`) once the canvas fills up (~15 paintings).
- Presentation (e.g. `format_summary`) lives in `gui/`, not the backend.
