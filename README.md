# 🖋️ Echoes of Expression: Creative AI Handwriting-to-Art Installation

This system transforms handwritten text into AI-generated artwork using emotion and cultural interpretation. A reactive agent system gradually draws the image onto a communal canvas, representing collective expression and interconnectedness.

---

## 🚀 How to Run the System

### 1. Open Terminal and Navigate to the Project Folder
```bash
cd [project-folder]
```

### 2. Start the Local LLM (Ollama)
The OCR, translation, emotion and cultural analysis run on a free local model through [Ollama](https://ollama.com).
```bash
ollama pull qwen2.5vl:7b
```
- Ollama runs in the background after installing; the app connects to it at `http://localhost:11434`.
- To use a different model or OpenAI instead, copy `.env.example` to `.env` and edit it.

### 3. Run the GUI Application
```bash
python main.py
```
- The upload button is enabled once the models have finished loading

To run the pipeline without the GUI:
```bash
python -m backend handwritingImages/handwriting1.png handwritingImages/handwriting6.png
```

### 4. Upload Handwriting Images
- Click the **"Upload Handwriting Images"** button in the UI.
- You can:
  - Select your own handwritten images, or  
  - Use samples from the `handwritingImages/` folder.
- Select up to 6 images at once; they are read in parallel and all painted onto the canvas at the same time.

### 5. Wait for Processing
- The system will interpret the text and emotion.
- This step may take a few seconds — progress is shown in the UI console.

### 6. Watch the Canvas Render
- Once processing is complete, the drawing begins automatically under the UI console.
- Each painting is drawn to completion before the upload button is enabled again.

### 7. View Full Results
- After rendering, the UI console displays all retrieved and generated data:
  - Recognized text
  - Emotion and cultural context
  - Generated prompt
  - Final image info

---

## ⚠️ Important Notes

- **One batch at a time.**  
  The upload button is disabled while a batch of images is being processed.

- **Language model**  
  By default the OCR and NLP steps use a local vision model (`qwen2.5vl:7b`) through Ollama, so no API key or credits are needed. Any OpenAI-compatible server works: set `LLM_BASE_URL`, `LLM_MODEL` and `LLM_API_KEY` in `.env` (see `.env.example`), e.g. to use OpenAI's GPT-4o.

- **System requirements**  
  Running this system locally requires Python 3.10 or later, at least 16GB of RAM, and a GPU with at least 8GB VRAM (12GB+ recommended, e.g. an NVIDIA RTX 3060 12GB or higher) to run Stable Diffusion (DreamShaper 8, an SD 1.5 fine-tune at 512x512, ~3.5GB VRAM) alongside the local language model. The first run downloads about 2GB of Stable Diffusion weights and 6GB for the Ollama model.

- **Requirements.txt**  
  Check the requirements.txt file to install all required libraries to run the system.
---

## 💾 Output

- The final full-size collaborative canvas will be saved at:  
  ```
  image/final_collaborative_canvas.png
  ```

---