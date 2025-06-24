import tkinter as tk
from tkinter import filedialog

X_PAD = 10
Y_PAD = 10

FONT_BIG = ("Helvetica", 14)
FONT_NORMAL = ("Helvetica", 12)


def run():
    print("Hello world")


def ask_file(extension, filetypes):
    filedialog.asksaveasfilename(
        defaultextension=extension,
        filetypes=filetypes,
        title="Save As"
    )


def main():
    root = tk.Tk()
    root.title("EM Utils")
    root.resizable(width=False, height=False)

    # Main label
    mainlabel = tk.Label(root, text="EM Utils", font=FONT_BIG)
    mainlabel.grid(row=0, column=0, columnspan=2, padx=X_PAD, pady=Y_PAD, )

    # Checkboxes

    # -Crop checkbox
    downsample_label = tk.Label(root, text="Downsample", font=FONT_NORMAL)


    downsample_entry = tk.Entry(root, font=FONT_NORMAL)
    downsample_entry.insert(0, "8")


    # Image path
    image_path_label = tk.Entry(root, font=FONT_NORMAL)
    image_path_label.insert(0, "No image folder selected")
    image_path_label.grid(row=3, column=1, columnspan=1, padx=X_PAD, pady=Y_PAD)

    image_path_button = tk.Button(
        root, text="Select Image Folder", command=lambda: ask_file(".dm3", [("Digital Micrograph Image", "*.dm3")]), font=FONT_NORMAL
    )
    image_path_button.grid(row=3, column=0, columnspan=1, padx=X_PAD, pady=Y_PAD)

    # Output path
    output_path_label = tk.Entry(root, font=FONT_NORMAL)
    output_path_label.insert(0, "No output folder selected")
    output_path_label.grid(row=4, column=1, columnspan=1, padx=X_PAD, pady=Y_PAD)

    output_path_button = tk.Button(
        root, text="Select Output Folder", command=lambda: ask_file(".tiff", [("TIFF File", "*.tif *.tiff")]), font=FONT_NORMAL
    )
    output_path_button.grid(row=4, column=0, columnspan=1, padx=X_PAD, pady=Y_PAD)

    # Run button
    run_button = tk.Button(
        root,
        text="Run!", width=30,
        command=lambda: run(
            image_path_label.get(), output_path_label.get(), crop_var.get(), adjust_brightness_var.get(),
            recursive_search_var.get(), output_label
        ),
        font=FONT_NORMAL
    )
    run_button.grid(row=5, column=0, columnspan=2, padx=X_PAD, pady=Y_PAD, )

    # Output label
    output_label = tk.Label(root, text="", font=FONT_NORMAL)
    output_label.grid(row=6, column=0, columnspan=2, padx=X_PAD, pady=Y_PAD)

    root.mainloop()


if __name__ == "__main__":
    main()
