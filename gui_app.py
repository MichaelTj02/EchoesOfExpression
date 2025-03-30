import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageOps
import threading
import os

# Import from project.py
from project import automate_from_image_file, canvas, CANVAS_WIDTH, CANVAS_HEIGHT

# ----------- GUI Setup -----------

class CulturalAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("✍️ Cultural AI Drawing System")
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
            self.output_text.insert(tk.END, f"🖼️ Selected: {file_path}\n")
            self.output_text.insert(tk.END, "🔄 Processing...\n")
            self.output_text.see(tk.END)

            threading.Thread(
                target=self.run_processing, 
                args=(file_path,), 
                daemon=True
            ).start()

    def update_canvas_preview(self, canvas):
        canvas.save("latest_canvas_preview.png")

        preview_img = Image.open("latest_canvas_preview.png").convert("RGBA")
        preview_img = ImageOps.contain(preview_img, (960, 540), method=Image.LANCZOS)
        preview_tk = ImageTk.PhotoImage(preview_img)
        self.canvas_preview_label.config(image=preview_tk)
        self.canvas_preview_label.image = preview_tk 

    def run_processing(self, image_path):
        try:
            summary, _ = automate_from_image_file(
                image_input=image_path, 
                update_callback=self.update_canvas_preview
            )
            self.output_text.insert(tk.END, f"\n✅ Drawing complete.\n\n{summary}")
            self.output_text.see(tk.END)
        except Exception as e:
            self.output_text.insert(tk.END, f"\n❌ Error: {e}\n")
            self.output_text.see(tk.END)
            messagebox.showerror("Processing Error", str(e))


# ----------- Run App -----------

if __name__ == "__main__":
    root = tk.Tk()
    app = CulturalAIGUI(root)
    root.mainloop()
