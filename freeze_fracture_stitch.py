from pathlib import Path
import stitch

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

import matplotlib.pyplot as plt


def crop_scale_bar(image):
    horizontal_scale_bar = image.shape[0] > image.shape[1]

    if horizontal_scale_bar:
        top_scale_bar_heuristic = np.count_nonzero(image[0] == 0)
        bottom_scale_bar_heuristic = np.count_nonzero(image[-1] == 0)

        top_scale_bar = top_scale_bar_heuristic > bottom_scale_bar_heuristic
        if top_scale_bar:
            image = image[image.shape[0] - image.shape[1]:, :]
        else:
            image = image[:image.shape[1], :]
    else:
        left_scale_bar_heuristic = np.count_nonzero(image[:, 0] == 0)
        right_scale_bar_heuristic = np.count_nonzero(image[:, -1] == 0)

        left_scale_bar = left_scale_bar_heuristic > right_scale_bar_heuristic
        if left_scale_bar:
            image = image[:, image.shape[1] - image.shape[0]:]
        else:
            image = image[:, :image.shape[0]]

    return image


def main():
    directory = Path('input/3')
    image_paths = list(directory.glob('*.tif')) + list(directory.glob('*.tiff'))

    images = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path))

        if img is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            cropped = crop_scale_bar(gray)
            bgr = cv2.cvtColor(cropped, cv2.COLOR_GRAY2BGR)  # Stitch expects BGR
            images.append(bgr)
            print(f"Loaded image {image_path}")
        else:
            print(f"Warning: Could not read image {image_path}")

    status, stitched = stitch.stitch(images, threshold=0.3, contrast=2.0)
    if status != cv2.Stitcher_OK:
        print(f"Stitching failed with status code: {status}")
        return

    cv2.imwrite("stitched_output.png", stitched)


if __name__ == "__main__":
    cv2.ocl.setUseOpenCL(False)
    main()
