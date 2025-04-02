# Echoes of Expression
IAT460 D101
Michael Tjokrowardojo (301416843)


# 🖋️ Creative AI Handwriting-to-Art Installation

This system transforms handwritten text into AI-generated artwork using emotion and cultural interpretation. A reactive agent system gradually draws the image onto a communal canvas, representing collective expression and interconnectedness.

---

## 🚀 How to Run the System

### 1. Open Terminal and Navigate to the Project Folder
```bash
cd [project-folder]
```

### 2. Run the GUI Application
```bash
python gui_app.py
```

### 3. Upload a Handwriting Image
- Click the **"Upload Handwriting Image"** button in the UI.
- You can:
  - Select your own handwritten image, or  
  - Use a sample from the `handwritingImages/` folder.

### 4. Wait for Processing
- The system will interpret the text and emotion.
- This step may take a few seconds — progress can be seen in the terminal.
- Seeing language not detected in the terminal is normal. (Unresolved bug)

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

- **Only input one image at a time.**  
  Submitting multiple images simultaneously may crash the system.

- **OpenAI API Key needed to run the system.**  
  The system relies on OpenAI's GPT-4o for its OCR and NLP system so a key is needed. Create a .env file in the folder and put your key inside.

- **System requirements**  
  Running this system locally requires Python 3.9 or later, at least 16GB of RAM, and a GPU with a minimum of 6GB VRAM (such as an NVIDIA RTX 2060 or higher) to support Stable Diffusion 2.1.
---

## 💾 Output

- The final full-size collaborative canvas will be saved at:  
  ```
  image/final_collaborative_canvas.png
  ```

---