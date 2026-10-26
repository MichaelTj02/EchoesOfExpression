import queue
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageOps, ImageTk

from backend import EchoesEngine, MissingAPIKeyError, PipelineResult

POLL_MS = 50
PREVIEW_SIZE = (960, 540)


def format_summary(result: PipelineResult) -> str:
    translated = result.translated_text if result.translated_text is not None else "Already in English"
    return (
        f"Text: {result.extracted_text}\n"
        f"Language: {result.language}\n"
        f"Translated: {translated}\n"
        f"Emotion: {result.emotion} (Confidence: {result.confidence:.2f})\n"
        f"Composition: {result.composition_style}\n"
        f"Painter: {result.painter or 'None suggested'}\n"
        f"Cultural Visual: {result.visual_goal}\n"
        f"Prompt: {result.prompt}"
    )


class CulturalAI:
    def __init__(self, root: tk.Tk, engine: EchoesEngine):
        self.root = root
        self.engine = engine
        self.root.title("🖌️ Echoes of Expression")
        self.root.geometry("1280x1024")

        # Worker threads never touch Tk; they post events here for _poll to handle
        self.events = queue.Queue()
        # Only the newest canvas frame matters, so frames overwrite a single slot instead of queueing
        self.latest_frame = None

        # Upload button (enabled once models are loaded)
        self.upload_btn = tk.Button(
            root,
            text="Upload Handwriting Image",
            command=self.upload_image,
            font=("Arial", 14),
            state=tk.DISABLED
        )
        self.upload_btn.pack(pady=10)

        # Text output area
        self.output_text = tk.Text(
            root, height=15, width=100,
            font=("Consolas", 10), wrap=tk.WORD,
            bg="#1e1e1e", fg="white", insertbackground="white"
        )
        self.output_text.pack(pady=10)

        # Canvas preview area
        self.canvas_preview_label = tk.Label(root)
        self.canvas_preview_label.pack(pady=10)

        self.root.after(POLL_MS, self._poll)
        self.start_loading()

    def log(self, message):
        self.output_text.insert(tk.END, f"{message}\n")
        self.output_text.see(tk.END)

    def start_loading(self):
        self.upload_btn.config(state=tk.DISABLED)
        self.log("Loading models...")
        threading.Thread(target=self._load_worker, daemon=True).start()

    def _load_worker(self):
        try:
            self.engine.load(on_progress=lambda stage, msg: self.events.put(("progress", stage, msg)))
            self.events.put(("loaded",))
        except MissingAPIKeyError:
            self.events.put(("need_key",))
        except Exception as e:
            self.events.put(("error", e))

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Handwriting Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.upload_btn.config(state=tk.DISABLED)
            self.output_text.delete("1.0", tk.END)
            self.log(f"Selected: {file_path}")
            self.log("Processing input...")

            threading.Thread(
                target=self._process_worker,
                args=(file_path,),
                daemon=True
            ).start()

    def _process_worker(self, image_path):
        try:
            result = self.engine.process(
                image_path,
                on_progress=lambda stage, msg: self.events.put(("progress", stage, msg)),
                on_frame=self._set_latest_frame,
            )
            self.events.put(("done", result))
        except Exception as e:
            self.events.put(("error", e))

    def _set_latest_frame(self, frame):
        self.latest_frame = frame  # attribute assignment is atomic; _poll picks it up

    def _poll(self):
        latest_frame, self.latest_frame = self.latest_frame, None
        try:
            while True:
                event = self.events.get_nowait()
                kind = event[0]
                if kind == "progress":
                    self.log(event[2])
                elif kind == "loaded":
                    self.upload_btn.config(state=tk.NORMAL)
                    self.update_canvas_preview(self.engine.canvas_snapshot())
                elif kind == "need_key":
                    self.ask_for_key()
                elif kind == "done":
                    self.log(f"\nDrawing complete.\n\n{format_summary(event[1])}")
                    self.upload_btn.config(state=tk.NORMAL)
                elif kind == "error":
                    self.log(f"\nError: {event[1]}")
                    messagebox.showerror("Processing Error", str(event[1]))
                    if self.engine.loaded:
                        self.upload_btn.config(state=tk.NORMAL)
        except queue.Empty:
            pass

        if latest_frame is not None:
            self.update_canvas_preview(latest_frame)

        self.root.after(POLL_MS, self._poll)

    def ask_for_key(self):
        key = simpledialog.askstring(
            "OpenAI API Key",
            "No OPENAI_API_KEY found in .env.\nEnter your OpenAI API key:",
            show="*",
            parent=self.root
        )
        if not key:
            self.log("No API key provided. Add OPENAI_API_KEY to a .env file and restart.")
            return
        self.engine = EchoesEngine(api_key=key.strip())
        self.start_loading()

    def update_canvas_preview(self, frame):
        resized = ImageOps.contain(frame, PREVIEW_SIZE, method=Image.LANCZOS)
        preview_tk = ImageTk.PhotoImage(resized)

        self.canvas_preview_label.config(image=preview_tk)
        self.canvas_preview_label.image = preview_tk


def run() -> None:
    root = tk.Tk()
    CulturalAI(root, EchoesEngine())
    root.mainloop()
