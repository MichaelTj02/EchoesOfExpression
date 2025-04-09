import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import threading
import time

# Import from project.py
from project import automate_from_image_file, canvas, CANVAS_WIDTH, CANVAS_HEIGHT

# ----------- GUI Setup -----------

class CulturalAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🖌️ Echoes of Expression")
        self.root.geometry("1280x1024")

        # Upload button
        self.upload_btn = tk.Button(
            root, 
            text="Upload Handwriting Image", 
            command=self.upload_image, 
            font=("Arial", 14)
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

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Handwriting Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert(tk.END, f"Selected: {file_path}\n")
            self.output_text.insert(tk.END, "Processing input...\n")
            self.output_text.see(tk.END)

            threading.Thread(
                target=self.run_processing, 
                args=(file_path,), 
                daemon=True
            ).start()

    def update_canvas_preview(self, canvas):
        # Save canvas temporarily
        canvas.save("image/latest_canvas_preview.png")
        time.sleep(0.05)  # Ensure file is written before loading

        # Open and fully load the canvas image
        with open("image/latest_canvas_preview.png", "rb") as f:
            preview_img = Image.open(f)
            preview_img.load()

        bg_color = (240, 230, 210, 255)
        background = Image.new("RGBA", preview_img.size, bg_color)
        composite = Image.alpha_composite(background, preview_img.convert("RGBA"))

        resized = ImageOps.contain(composite, (960, 540), method=Image.LANCZOS)
        preview_tk = ImageTk.PhotoImage(resized)

        # Update GUI
        self.canvas_preview_label.config(image=preview_tk)
        self.canvas_preview_label.image = preview_tk

    def run_processing(self, image_path):
        try:
            summary, _ = automate_from_image_file(
                image_input=image_path, 
                update_callback=self.update_canvas_preview
            )
            self.output_text.insert(tk.END, f"\nDrawing complete.\n\n{summary}")
            self.output_text.see(tk.END)
        except Exception as e:
            self.output_text.insert(tk.END, f"\nError: {e}\n")
            self.output_text.see(tk.END)
            messagebox.showerror("Processing Error", str(e))


# ----------- Run App -----------

if __name__ == "__main__":
    root = tk.Tk()
    app = CulturalAIGUI(root)
    root.mainloop()
