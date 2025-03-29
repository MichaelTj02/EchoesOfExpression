import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import threading
import os

# Import your automation method
from project import automate_from_image_file, canvas

# ----------- GUI Setup -----------

class CulturalAIGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("✍️ Cultural AI Drawing System")
        self.root.geometry("800x600")

        # Upload button
        self.upload_btn = tk.Button(root, text="Upload Handwriting Image", command=self.upload_image, font=("Arial", 14))
        self.upload_btn.pack(pady=20)

        # Text output area
        self.output_text = tk.Text(root, height=15, width=90, font=("Consolas", 10))
        self.output_text.pack(pady=10)

        # Canvas preview label
        self.canvas_preview_label = tk.Label(root)
        self.canvas_preview_label.pack()

    def upload_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Handwriting Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )

        if file_path:
            self.output_text.insert(tk.END, f"\nSelected: {file_path}\n")
            self.output_text.insert(tk.END, "Processing...\n")
            self.output_text.see(tk.END)

            # Run in background to keep UI responsive
            threading.Thread(target=self.run_processing, args=(file_path,), daemon=True).start()

    def update_canvas_preview(self, canvas):
        canvas.save("latest_canvas_preview.png")
        preview_img = Image.open("latest_canvas_preview.png").resize((500, 300))
        preview_tk = ImageTk.PhotoImage(preview_img)
        self.canvas_preview_label.config(image=preview_tk)
        self.canvas_preview_label.image = preview_tk  # Keep reference

    def run_processing(self, image_path):
        try:
            summary, _ = automate_from_image_file(image_path, update_callback=self.update_canvas_preview)

            self.output_text.insert(tk.END, f"\n{summary}\nDrawing complete.\n")
            self.output_text.see(tk.END)

        except Exception as e:
            messagebox.showerror("Error", str(e))
            self.output_text.insert(tk.END, f"\nError: {e}\n")
            self.output_text.see(tk.END)

# ----------- Run App -----------

if __name__ == "__main__":
    root = tk.Tk()
    app = CulturalAIGUI(root)
    root.mainloop()
