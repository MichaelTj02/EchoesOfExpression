import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

from PIL import Image, ImageOps, ImageTk

from backend import EchoesEngine, MissingAPIKeyError, PipelineResult
from backend.config import MAX_UPLOADS_PER_BATCH

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


def format_batch_summary(results, paths) -> str:
    """One section per uploaded image: its summary, or the error that stopped it."""
    sections = []
    for index, (path, result) in enumerate(zip(paths, results), start=1):
        header = f"[{index}/{len(results)}] {Path(path).name}"
        if isinstance(result, Exception):
            sections.append(f"{header}\nError: {result}")
        else:
            sections.append(f"{header}\n{format_summary(result)}")
    return "\n\n".join(sections)


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
        # Uploads can overlap (a new one may start while earlier ones are still drawing), so each gets a number
        self.upload_count = 0

        # Upload button (enabled once models are loaded)
        self.upload_btn = tk.Button(
            root,
            text="Upload Handwriting Images",
            command=self.upload_images,
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

    def upload_images(self):
        file_paths = filedialog.askopenfilenames(
            title=f"Select Handwriting Images (up to {MAX_UPLOADS_PER_BATCH})",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if not file_paths:
            return
        if len(file_paths) > MAX_UPLOADS_PER_BATCH:
            messagebox.showwarning(
                "Too Many Images",
                f"Please select at most {MAX_UPLOADS_PER_BATCH} images at once "
                f"({len(file_paths)} selected)."
            )
            return

        # The button stays enabled: the canvas keeps drawing earlier uploads while this one is processed
        file_paths = list(file_paths)
        self.upload_count += 1
        tag = f"[Upload {self.upload_count}]"
        self.log(f"\n{tag} Selected: {', '.join(Path(path).name for path in file_paths)}")

        threading.Thread(
            target=self._process_worker,
            args=(file_paths, tag),
            daemon=True
        ).start()

    def _process_worker(self, image_paths, tag):
        try:
            results = self.engine.process_many(
                image_paths,
                on_progress=lambda stage, msg: self.events.put(("progress", stage, f"{tag} {msg}")),
                on_frame=self._set_latest_frame,
            )
            self.events.put(("done", results, image_paths, tag))
        except Exception as e:
            self.events.put(("error", e, tag))

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
                    results, paths, tag = event[1], event[2], event[3]
                    if all(isinstance(result, Exception) for result in results):
                        self.log(f"\n{tag} No image could be processed.")
                    else:
                        self.log(f"\n{tag} Drawing complete.")
                    self.log(f"\n{format_batch_summary(results, paths)}")
                elif kind == "error":
                    tag = f"{event[2]} " if len(event) > 2 else ""  # load errors have no upload tag
                    self.log(f"\n{tag}Error: {event[1]}")
                    messagebox.showerror("Processing Error", str(event[1]))
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
