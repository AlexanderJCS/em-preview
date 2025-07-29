import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
import threading
import os
from pathlib import Path

# Import stitch_images from stitch.py
from stitch import load_and_stitch

def add_input_files():
    file_paths = filedialog.askopenfilenames(
        title="Select Input Images",
        filetypes=[("Image files", "*.dm3 *.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
    )

    all_files_to_add = []

    for p in file_paths:
        all_files_to_add.append(p)

        # If "add similar files" is checked, look for files with same name in sibling directories
        if add_similar_var.get():
            path = Path(p)
            filename = path.name
            parent_dir = path.parent
            grandparent_dir = parent_dir.parent

            # Check if we can go up a directory
            if grandparent_dir.exists() and grandparent_dir != parent_dir:
                # Find all sibling directories
                for sibling_dir in grandparent_dir.iterdir():
                    if (sibling_dir.is_dir() and
                        sibling_dir != parent_dir and
                        sibling_dir.name != parent_dir.name):

                        # Look for file with same name in sibling directory (not recursive)
                        potential_file = sibling_dir / filename
                        if potential_file.exists() and potential_file.is_file():
                            all_files_to_add.append(str(potential_file))

    # Add all files to listbox (avoiding duplicates)
    for file_path in all_files_to_add:
        if file_path not in input_listbox.get(0, tk.END):
            input_listbox.insert(tk.END, file_path)

def remove_selected_files():
    for idx in reversed(input_listbox.curselection()):
        input_listbox.delete(idx)

def clear_input_files():
    input_listbox.delete(0, tk.END)

def browse_output_file():
    output_file = filedialog.asksaveasfilename(
        title="Select Stitched Output File",
        defaultextension=".jpg",
        filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("TIFF", "*.tiff"), ("All files", "*.*")]
    )
    if output_file:
        output_file_var.set(output_file)

def browse_downscaled_output():
    output_dir = filedialog.askdirectory(
        title="Select Downscaled Output Directory"
    )
    if output_dir:
        downscaled_output_var.set(output_dir)

def do_stitching():
    try:
        input_files     = input_listbox.get(0, tk.END)
        threshold       = float(threshold_var.get())
        downscaling     = float(downscaling_var.get())
        output_file     = output_file_var.get()
        downscaled_output = downscaled_output_var.get()
        contrast        = float(contrast_var.get())
        invert          = invert_var.get()

        result = load_and_stitch(
            image_paths=list(input_files),
            output_path=output_file,
            threshold=threshold,
            downscaling=downscaling,
            contrast=contrast,
            downscaled_output_dir=downscaled_output if downscaled_output else None,
            invert=invert,
            show_preview=show_preview_var.get(),
            preview_width=int(preview_width_var.get()) if preview_width_var.get().isdigit() else 2
        )

        # back on the main thread: re-enable button and show result
        def on_done():
            run_button.config(state=tk.NORMAL)
            if result == 0:
                success_msg = f"Stitched image saved to:\n{output_file}"
                if downscaled_output:
                    success_msg += f"\n\nDownscaled images saved to downscaled output directory"
                messagebox.showinfo("Success", success_msg)
            else:
                msg = f"Stitching failed with status code: {result}"
                if result == 1:
                    msg = "Stitching failed: Not enough overlap"
                elif result == 2:
                    msg = "Stitching failed: Could not compute perspective transform"
                elif result == 3:
                    msg = "Stitching failed: Not camera setting failed"
                messagebox.showerror("Error", msg)
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
contrast_var     = tk.StringVar(value="2.0")
output_file_var  = tk.StringVar()
downscaled_output_var = tk.StringVar()
add_similar_var  = tk.BooleanVar(value=True)
invert_var       = tk.BooleanVar(value=True)
show_preview_var = tk.BooleanVar(value=False)
preview_width_var = tk.StringVar(value="2")

# Layout
ttk.Label(root, text="Input Images:").grid(row=0, column=0, sticky="nw", padx=10, pady=10)
input_listbox = tk.Listbox(root, selectmode=tk.EXTENDED, width=80, height=10)
input_listbox.grid(row=0, column=1, columnspan=2, padx=10, pady=10, sticky="nsew")

btn_frame = ttk.Frame(root)
btn_frame.grid(row=1, column=1, columnspan=2, sticky="w", padx=10)
ttk.Button(btn_frame, text="Add Files…",    command=add_input_files).grid(row=0, column=0, padx=(0,5))
ttk.Button(btn_frame, text="Remove Files…", command=remove_selected_files).grid(row=0, column=1, padx=(0,5))
ttk.Button(btn_frame, text="Clear List",    command=clear_input_files).grid(row=0, column=2)

# Checkbox for adding similar files (above Confidence row)
ttk.Checkbutton(root, text="Add similar files", variable=add_similar_var).grid(row=2, column=1, columnspan=2, sticky="w", padx=10, pady=5)

ttk.Label(root, text="Confidence:").grid(row=3, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=threshold_var).grid(row=3, column=1, sticky="we", padx=10)

ttk.Label(root, text="Downscaling:").grid(row=4, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=downscaling_var).grid(row=4, column=1, sticky="we", padx=10)

ttk.Label(root, text="Contrast (lower=higher contrast):").grid(row=5, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=contrast_var).grid(row=5, column=1, sticky="we", padx=10)

ttk.Label(root, text="Downscaled Output:").grid(row=6, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=downscaled_output_var, width=60).grid(row=6, column=1, padx=10, pady=10, sticky="we")
ttk.Button(root, text="Browse…", command=browse_downscaled_output).grid(row=6, column=2, padx=10)

ttk.Label(root, text="Stitched Output:").grid(row=7, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=output_file_var, width=60).grid(row=7, column=1, padx=10, pady=10, sticky="we")
ttk.Button(root, text="Browse…", command=browse_output_file).grid(row=7, column=2, padx=10)

# Checkbox for inverting image (above Run Stitching button)
ttk.Checkbutton(root, text="Invert image", variable=invert_var).grid(row=8, column=1, columnspan=2, sticky="w", padx=10, pady=5)

# Checkbox for showing preview
ttk.Checkbutton(root, text="Show Preview", variable=show_preview_var).grid(row=9, column=1, columnspan=2, sticky="w", padx=10, pady=5)

# Preview Image Width input
ttk.Label(root, text="Preview Image Width:").grid(row=10, column=0, sticky="e", padx=10, pady=10)
ttk.Entry(root, textvariable=preview_width_var).grid(row=10, column=1, sticky="we", padx=10)

run_button = ttk.Button(root, text="Run Stitching", command=start_stitching)
run_button.grid(row=11, column=1, pady=20)

# Make the listbox expand with the window
root.columnconfigure(1, weight=1)
root.rowconfigure(0, weight=1)

root.mainloop()
