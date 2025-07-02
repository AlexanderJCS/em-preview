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

import dm3_lib as dm3

# TODO: Check OpenCV version >= 3.4


def preprocess_image(image, downscaling=8):
    downscaled = cv2.resize(image, (0, 0), fx=1 / downscaling, fy=1 / downscaling, interpolation=cv2.INTER_AREA)

    lower_percentile = np.percentile(downscaled[:, :, 0], 5)
    upper_percentile = np.percentile(downscaled[:, :, 0], 95)

    return downscaled, lower_percentile, upper_percentile


def stitch_images(image_paths, output_path='panorama.jpg', threshold=0.5, downscaling=8, contrast=2.0):
    # Load all images
    images = []

    lower_percentiles = []
    upper_percentiles = []

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
                lower_percentiles.append(lower_percentile)
                upper_percentiles.append(upper_percentile)
        else:
            img = cv2.imread(p)
            if img is None:
                print(f"Error: could not read image {p}")
                return False

            downscaled, lower_percentile, upper_percentile = preprocess_image(img, downscaling)
            images.append(downscaled)
            lower_percentiles.append(lower_percentile)
            upper_percentiles.append(upper_percentile)

    # Construct histogram for the GMM
    values = []
    for img in images:
        values.append(img[:, :, 0].flatten())

    values = np.array(values).flatten()

    # n_components = 3
    # gmm = GaussianMixture(n_components=n_components, covariance_type='full')
    # gmm.fit(values.reshape(-1, 1))
    #
    # print("GMM means:", gmm.means_.flatten())
    # print("GMM covariances:", gmm.covariances_.flatten())
    #
    # lower_clip = np.min(lower_percentiles)
    # upper_clip = np.max(upper_percentiles)

    mean = values.mean()
    std = values.std()

    lower_clip = mean - contrast * std
    upper_clip = mean + contrast * std

    # for i, img in enumerate(images):
    #     hist, bins = np.histogram(img[:, :, 0], bins=100)
    #     plt.plot(bins[:-1], hist, label=f'Histogram {i + 1}')
    # plt.xlabel('Pixel Intensity')
    # plt.ylabel('Frequency')
    # plt.axvline(float(lower_clip), color='r', linestyle='--', label='Lower Cutoff')
    # plt.axvline(float(upper_clip), color='g', linestyle='--', label='Upper Cutoff')
    # plt.legend()
    # plt.show()

    # Normalize images to the same range
    for i, img in enumerate(images):
        print(f"Normalizing image {i + 1} of {len(images)}")
        # Clip to the percentiles
        img = np.clip(img, lower_clip, upper_clip)
        # Normalize to 0-255
        img = ((img - lower_clip) / (upper_clip - lower_clip) * 255).astype(np.uint8)
        images[i] = img

        # cv2.imshow("i", img)
        # cv2.waitKey(0)

    for i, img in enumerate(images):
        print(f"Image {i + 1} shape after resize: {img.shape}")
        if img.size == 0:
            raise RuntimeError(f"Image {i + 1} is empty after resizing")

    print("Creating stitcher...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    stitcher.setPanoConfidenceThresh(threshold)

    print("Stitching... this may take a while")
    status, pano = stitcher.stitch(images)
    # status == 0 means OK
    if status != 0:
        print(f"Stitching failed (status code {status})")
        return status

    print("Writing stitched image...")
    cv2.imwrite(output_path, pano)
    print(f"Stitched image saved to {output_path}")
    return 0


def main():
    start = time.perf_counter()

    directory = Path("input/2")
    files = [str(f) for f in directory.iterdir() if f.is_file()]
    result = stitch_images(files, "out.tif", threshold=0.4, downscaling=4)
    if result != 0:
        sys.exit(1)

    end = time.perf_counter()
    print(f"Finished in {end - start} seconds")


if __name__ == '__main__':
    main()
