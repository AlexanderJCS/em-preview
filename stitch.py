from pathlib import Path
import time
import sys
from datetime import datetime

try:
    import cv2
except ImportError:
    print("OpenCV not found. Please run:\npip install opencv-python\nto install OpenCV")
    exit()

try:
    import numpy as np
except ImportError:
    print("Numpy not found. Please run:\npip install numpy\n to install numpy")
    exit()

import dm3_lib as dm3

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
except ImportError:
    print("Matplotlib not found. Preview functionality will be disabled.")
    plt = None

# TODO: Check OpenCV version >= 3.4


def preprocess_image(image, downscaling=8):
    downscaled = cv2.resize(image, (0, 0), fx=1 / downscaling, fy=1 / downscaling, interpolation=cv2.INTER_AREA)

    lower_percentile = np.percentile(downscaled[:, :, 0], 5)
    upper_percentile = np.percentile(downscaled[:, :, 0], 95)

    return downscaled, lower_percentile, upper_percentile


def stitch(images, threshold=0.5, contrast=2.0, downscaled_output_dir=None, original_paths=None, invert=True, show_preview=False, preview_width=2):
    # Construct histogram for the GMM
    values = []
    for img in images:
        values.append(img[:, :, 0].flatten())

    values = np.array(values).flatten()

    mean = values.mean()
    std = values.std()

    lower_clip = mean - contrast * std
    upper_clip = mean + contrast * std

    # Normalize images to the same range
    for i, img in enumerate(images):
        # Clip to the percentiles
        img = np.clip(img, lower_clip, upper_clip)
        # Normalize to 0-255
        img = ((img - lower_clip) / (upper_clip - lower_clip) * 255).astype(np.uint8)
        images[i] = img

    # Save normalized downscaled images if output directory is specified
    if downscaled_output_dir and original_paths:
        save_downscaled_images(images, original_paths, downscaled_output_dir, invert)

    # Show preview if requested, after downscaling but before stitching
    if show_preview and images:
        show_preview_grid(images, preview_width)

    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    stitcher.setPanoConfidenceThresh(threshold)

    status, pano = stitcher.stitch(images)
    return status, pano


def load_and_stitch(image_paths, output_path='panorama.jpg', threshold=0.5, downscaling=8, contrast=2.0, downscaled_output_dir=None, invert=True, show_preview=False, preview_width=2):
    # Load all images
    images = []

    for i, p in enumerate(image_paths):
        print(f"Loading image {i + 1} of {len(image_paths)}")

        # Check if extension ends in dm3
        if p.split(".")[-1] == "dm3":
            dm3f = dm3.DM3(p)
            img = dm3f.imagedata
            # Iterate through all images in the stack if there are several
            print(img.shape, img.ndim)
            if img.ndim != 3:
                img = np.array([img])

            for j in range(img.shape[0]):
                gray = img[j, :, :]  # Take the image in the stack
                # Convert to 3-channel image by duplicating the single channel
                color = np.stack((gray, gray, gray), axis=-1)
                downscaled, lower_percentile, upper_percentile = preprocess_image(color, downscaling)
                images.append(downscaled)
        else:
            img = cv2.imread(p)
            if img is None:
                print(f"Error: could not read image {p}")
                return False

            downscaled, lower_percentile, upper_percentile = preprocess_image(img, downscaling)
            images.append(downscaled)

    status, pano = stitch(images, threshold=threshold, contrast=contrast, downscaled_output_dir=downscaled_output_dir, original_paths=image_paths, invert=invert, show_preview=show_preview, preview_width=preview_width)
    if status == cv2.Stitcher_OK:
        print("Stitching completed successfully.")
        # Conditionally invert the image based on the invert parameter
        if invert:
            cv2.imwrite(output_path, cv2.bitwise_not(pano))
        else:
            cv2.imwrite(output_path, pano)

    return status


def save_downscaled_images(downscaled_images, original_paths, output_dir, invert=True):
    """Save downscaled images to specified directory with timestamp and first file name"""
    if not downscaled_images or not original_paths:
        return

    # Get first file name (without extension and path)
    first_file = Path(original_paths[0]).stem

    # Create timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Create directory name
    dir_name = f"downscaled_{timestamp}_{first_file}"
    full_output_dir = Path(output_dir) / dir_name

    # Create directory
    full_output_dir.mkdir(parents=True, exist_ok=True)

    # Save each downscaled image
    for i, img in enumerate(downscaled_images, 1):
        output_filename = f"downscaled_{i}.jpg"
        output_path = full_output_dir / output_filename
        # Conditionally invert the image based on the invert parameter
        if invert:
            cv2.imwrite(str(output_path), cv2.bitwise_not(img))
        else:
            cv2.imwrite(str(output_path), img)
        print(f"Saved downscaled image: {output_path}")

    print(f"All downscaled images saved to: {full_output_dir}")


def show_preview_grid(images, preview_width):
    """Display images in a grid using matplotlib with no gaps or text"""
    if plt is None:
        print("Matplotlib not available, skipping preview")
        return

    num_images = len(images)
    num_rows = (num_images + preview_width - 1) // preview_width  # Ceiling division

    # Create figure with no spacing between subplots
    fig, axes = plt.subplots(num_rows, preview_width, figsize=(preview_width * 4, num_rows * 4))
    fig.subplots_adjust(wspace=0, hspace=0)  # Remove gaps between images

    # Handle case where there's only one subplot
    if num_rows == 1 and preview_width == 1:
        axes = [axes]
    elif num_rows == 1:
        axes = [axes]
    elif preview_width == 1:
        axes = [[ax] for ax in axes]
    else:
        # Convert to 2D array if needed
        if len(axes.shape) == 1:
            axes = axes.reshape(num_rows, preview_width)

    for i in range(num_rows * preview_width):
        row = i // preview_width
        col = i % preview_width

        if num_rows == 1 and preview_width == 1:
            ax = axes[0]
        elif num_rows == 1:
            ax = axes[col]
        elif preview_width == 1:
            ax = axes[row][0]
        else:
            ax = axes[row][col]

        if i < num_images:
            # Get the image and ensure it's in the right format
            img = images[i].copy()

            # Convert to grayscale if it's a 3-channel image with identical channels
            if len(img.shape) == 3:
                # Check if it's actually grayscale (all channels identical)
                if np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,1], img[:,:,2]):
                    img = img[:,:,0]  # Use single channel
                else:
                    # Convert BGR to RGB for matplotlib
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Ensure proper data type and range
            if img.dtype != np.uint8:
                img = img.astype(np.uint8)

            # Display image
            ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None, vmin=0, vmax=255)
        # else:
        #     # Fill empty spaces with black
        #     if len(images) > 0:
        #         black_img = np.zeros_like(images[0][:,:,0] if len(images[0].shape) == 3 else images[0])
        #     else:
        #         black_img = np.zeros((100, 100))
        #     ax.imshow(black_img, cmap='gray', vmin=0, vmax=255)

        # Remove all axes decorations
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')

    plt.tight_layout(pad=0)  # Remove padding
    plt.show()


def main():
    start = time.perf_counter()

    directory = Path("input/2")
    files = [str(f) for f in directory.iterdir() if f.is_file()]
    result = load_and_stitch(files, "out.tif", threshold=0.4, downscaling=4)
    if result != 0:
        sys.exit(1)

    end = time.perf_counter()
    print(f"Finished in {end - start} seconds")


if __name__ == '__main__':
    main()
