from pathlib import Path
import time
import sys

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

from packaging import version


if version.parse(cv2.__version__) < version.parse("3.4"):
    raise RuntimeError("OpenCV version 3.4 or higher is required for stitching. Please update OpenCV.")


def stitch_images(image_paths, output_path='panorama.jpg'):
    # Load all images
    images = []

    min_lower_percentile = np.inf
    max_upper_percentile = -np.inf

    for i, p in enumerate(image_paths):
        print(f"Loading image {i + 1} of {len(image_paths)}")

        img = cv2.imread(p)
        if img is None:
            print(f"Error: could not read image {p}")
            return False

        downscaled = cv2.resize(img, (0, 0), fx=0.125, fy=0.125, interpolation=cv2.INTER_AREA)

        lower_percentile = np.percentile(downscaled[:, :, 0], 1)
        upper_percentile = np.percentile(downscaled[:, :, 0], 99)

        min_lower_percentile = min(min_lower_percentile, lower_percentile)
        max_upper_percentile = max(max_upper_percentile, upper_percentile)

        images.append(downscaled)

    print(min_lower_percentile, max_upper_percentile)

    # Normalize images to the same range
    for i, img in enumerate(images):
        print(f"Normalizing image {i + 1} of {len(images)}")
        # Clip to the percentiles
        img = np.clip(img, min_lower_percentile, max_upper_percentile)
        # Normalize to 0-255
        img = ((img - min_lower_percentile) / (max_upper_percentile - min_lower_percentile) * 255).astype(np.uint8)
        images[i] = img

        # cv2.imshow("stitched image", img)
        # cv2.waitKey(0)

    for i, img in enumerate(images):
        print(f"Image {i + 1} shape after resize: {img.shape}")
        if img.size == 0:
            raise RuntimeError(f"Image {i + 1} is empty after resizing")

    print("Creating stitcher...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    stitcher.setPanoConfidenceThresh(0.5)

    print("Stitching... this may take a while")
    status, pano = stitcher.stitch(images)
    # status == 0 means OK
    if status != 0:
        print(f"Stitching failed (status code {status})")
        return False

    print("Writing stitched image...")
    cv2.imwrite(output_path, pano)
    print(f"Stitched image saved to {output_path}")
    return True


def main():
    start = time.perf_counter()

    directory = Path("input")
    files = [f for f in directory.iterdir() if f.is_file()]
    success = stitch_images(files, "out.tif")
    if not success:
        sys.exit(1)

    end = time.perf_counter()
    print(f"Finished in {end - start} seconds")


if __name__ == '__main__':
    main()
