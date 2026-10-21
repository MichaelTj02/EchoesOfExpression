# 🖋️ Echoes of Expression: Creative AI Handwriting-to-Art Installation

This system transforms handwritten text into AI-generated artwork using emotion and cultural interpretation. A reactive agent system gradually draws the image onto a communal canvas, representing collective expression and interconnectedness.

---

## 🚀 How to Run the System

### 1. Open Terminal and Navigate to the Project Folder
```bash
cd [project-folder]
```

### 2. Run the GUI Application & Enter OpenAI API Key
```bash
python main.py
```
- If no `.env` file with `OPENAI_API_KEY` is found, the app asks for your OpenAI API Key in a dialog
- The upload button is enabled once the models have finished loading

To run the pipeline without the GUI:
```bash
python -m backend handwritingImages/handwriting1.png
```

### 3. Upload a Handwriting Image
- Click the **"Upload Handwriting Image"** button in the UI.
- You can:
  - Select your own handwritten image, or  
  - Use a sample from the `handwritingImages/` folder.

### 4. Wait for Processing
- The system will interpret the text and emotion.
- This step may take a few seconds — progress is shown in the UI console.

### 5. Watch the Canvas Render
- Once processing is complete, the drawing begins automatically under the UI console.

### 6. View Full Results
- After rendering, the UI console displays all retrieved and generated data:
  - Recognized text
  - Emotion and cultural context
  - Generated prompt
  - Final image info

---

## ⚠️ Important Notes

- **One image at a time.**  
  The upload button is disabled while an image is being processed.

- **OpenAI API Key needed to run the system.**  
  The system relies on OpenAI's GPT-4o for its OCR and NLP system so a key is needed. Create a .env file in the folder and put your key inside.

- **System requirements**  
  Running this system locally requires Python 3.9 or later, at least 16GB of RAM, and a GPU with a minimum of 6GB VRAM (such as an NVIDIA RTX 2060 or higher) to support Stable Diffusion 2.1.

- **Requirements.txt**  
  Check the requirements.txt file to install all required libraries to run the system.
---

## 💾 Output

- The final full-size collaborative canvas will be saved at:  
  ```
  image/final_collaborative_canvas.png
  ```

---