import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import os

# Import stitch_images from stitch.py
from stitch import stitch_images

def add_input_files():
    file_paths = filedialog.askopenfilenames(
        title="Select Input Images",
        filetypes=[("Image files", "*.dm3 *.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
    )
    for p in file_paths:
        if p not in input_listbox.get(0, tk.END):
            input_listbox.insert(tk.END, p)

def remove_selected_files():
    for idx in reversed(input_listbox.curselection()):
        input_listbox.delete(idx)

def clear_input_files():
    input_listbox.delete(0, tk.END)

def browse_output_file():
    output_file = filedialog.asksaveasfilename(
        title="Select Output File",
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tiff"), ("All files", "*.*")]
    )
    if output_file:
        output_file_var.set(output_file)

def do_stitching():
    try:
        input_files     = input_listbox.get(0, tk.END)
        threshold       = float(threshold_var.get())
        downscaling     = float(downscaling_var.get())
        output_file     = output_file_var.get()

        success = stitch_images(
            image_paths=list(input_files),
            output_path=output_file,
            threshold=threshold,
            downscaling=downscaling
        )

        # back on the main thread: re-enable button and show result
        def on_done():
            run_button.config(state=tk.NORMAL)
            if success:
                messagebox.showinfo("Success", f"Stitched image saved to:\n{output_file}")
            else:
                messagebox.showerror("Error", "Stitching failed.")
        root.after(0, on_done)

    except Exception as e:
        def on_error():
            run_button.config(state=tk.NORMAL)
            messagebox.showerror("Exception", str(e))
        root.after(0, on_error)

def start_stitching():
    # Basic validation before threading
    if not input_listbox.get(0, tk.END) or not output_file_var.get():
        messagebox.showerror("Missing Input", "Please specify input files and output file.")
        return

    if not all(os.path.isfile(p) for p in input_listbox.get(0, tk.END)):
        messagebox.showerror("Invalid Files", "One or more input files are invalid.")
        return

    # Disable the run button and start background thread
    run_button.config(state=tk.DISABLED)
    threading.Thread(target=do_stitching, daemon=True).start()

# --- GUI setup ---
root = tk.Tk()
root.title("Image Stitcher")
root.geometry("900x600")
root.minsize(800, 500)

# Variables
threshold_var    = tk.StringVar(value="0.5")
downscaling_var  = tk.StringVar(value="8")
output_file_var  = tk.StringVar()

# Layout
ttk.Label(root, text="Input Images:").grid(row=0, column=0, sticky="nw", padx=10, pady=10)
input_listbox = tk.Listbox(root, selectmode=tk.EXTENDED, width=80, height=10)
input_listbox.grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky="nsew")

btn_frame = ttk.Frame(root)
btn_frame.grid(row=1, column=1, columnspan=2, sticky="w", padx=10)
ttk.Button(btn_frame, text="Add Files…",    command=add_input_files).grid(row=0, column=0, padx=(0,5))
ttk.Button(btn_frame, text="Remove Files…", command=remove_selected_files).grid(row=0, column=1, padx=(0,5))
ttk.Button(btn_frame, text="Clear List",    command=clear_input_files).grid(row=0, column=2)

ttk.Label(root, text="Threshold:").grid(row=2, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=threshold_var).grid(row=2, column=1, sticky="we", padx=10)

ttk.Label(root, text="Downscaling:").grid(row=3, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=downscaling_var).grid(row=3, column=1, sticky="we", padx=10)

ttk.Label(root, text="Output File:").grid(row=4, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=output_file_var, width=60).grid(row=4, column=1, padx=10, pady=10, sticky="we")
ttk.Button(root, text="Browse…", command=browse_output_file).grid(row=4, column=2, padx=10)

run_button = ttk.Button(root, text="Run Stitching", command=start_stitching)
run_button.grid(row=5, column=1, pady=20)

# Make the listbox expand with the window
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)

root.mainloop()
