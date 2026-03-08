import cv2
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Load image
# ──────────────────────────────────────────────────────────────────────────────
image = cv2.imread('iron_man_noisy.jpg')
if image is None:
    raise FileNotFoundError("Could not load 'iron_man_noisy.jpg'. Make sure it is in the working directory.")
image_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# ──────────────────────────────────────────────────────────────────────────────
# Method 1 – Custom kernel filtering + thresholding + Canny edge removal
#
# Steps:
#   1. Apply a custom 5×5 weighted-average kernel to smooth/denoise the image.
#   2. Threshold the filtered image to binary (pixel > 90 → 255, else 0).
#   3. Detect edges with Canny and remove them from the binary image.
# ──────────────────────────────────────────────────────────────────────────────
def method_1(image):
    """Clean the image using a custom convolution kernel, binary thresholding,
    and Canny edge removal.  Returns the binary result."""

    # Custom 5×5 smoothing kernel (weights sum to 25, then normalised)
    kernel = np.array([
        [0,    1,    1,    1,    0],
        [1,    1.25, 2,    1.25, 1],
        [1,    2,    0,    2,    1],
        [1,    1.25, 2,    1.25, 1],
        [0,    1,    1,    1,    0]
    ]) / 25

    # Convolve the image with the kernel
    filtered_image = cv2.filter2D(image, -1, kernel)

    # Binary threshold at intensity 90
    threshold = 90
    binary_image = np.where(filtered_image > threshold, 255, 0).astype(np.uint8)

    # Detect edges with Canny and remove them from the binary image
    edges_canny = cv2.Canny(binary_image, 100, 200)
    binary_image[edges_canny > 0] = 0

    # Save outputs
    cv2.imwrite('method1_filtered.jpg', filtered_image)
    cv2.imwrite('method1_binary.jpg', binary_image)
    cv2.imwrite('method1_edges_canny.jpg', edges_canny)
    print("[Method 1] Saved: method1_filtered.jpg, method1_binary.jpg, method1_edges_canny.jpg")

    return binary_image


# ──────────────────────────────────────────────────────────────────────────────
# Method 2 – Otsu binarisation + connected-component cluster removal
#
# Steps:
#   1. Convert to grayscale (if needed) and apply Otsu's automatic threshold
#      to produce a clean binary image.
#   2. Find all connected components (8-connectivity).
#   3. Remove any cluster (connected region) with fewer than 7 pixels.
# ──────────────────────────────────────────────────────────────────────────────
def method_2(image, min_cluster_size=7):
    """Clean the image using Otsu's binarisation and removal of small
    connected-component clusters.  Returns the cleaned binary result."""

    # Convert to grayscale if the image is colour
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Otsu's thresholding automatically picks the optimal threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Label every connected component (8-connectivity includes diagonals)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # Remove clusters smaller than min_cluster_size
    cleaned = binary.copy()
    removed_count = 0
    for label in range(1, num_labels):          # label 0 is background
        area = stats[label, cv2.CC_STAT_AREA]
        if area < min_cluster_size:
            cleaned[labels == label] = 0
            removed_count += 1

    # Save output
    cv2.imwrite('method2_cleaned.jpg', cleaned)
    print(f"[Method 2] Removed {removed_count} clusters with < {min_cluster_size} points. Saved: method2_cleaned.jpg")

    return binary, cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Run both methods and display a side-by-side comparison
# ──────────────────────────────────────────────────────────────────────────────
def compare_methods(image):
    """Run Method 1 and Method 2 on the same image and show results side by side."""

    print("=" * 60)
    print("Running Method 1 (kernel filter + Canny edge removal)...")
    print("=" * 60)
    result_m1 = method_1(image)

    print()
    print("=" * 60)
    print("Running Method 2 (Otsu binarisation + cluster removal)...")
    print("=" * 60)
    binary_m2, cleaned_m2 = method_2(image)

    # --- Build comparison display ---
    # Convert all single-channel results to BGR so they can be stacked with the colour original
    original_resized = image.copy()
    m1_bgr = cv2.cvtColor(result_m1, cv2.COLOR_GRAY2BGR) if len(result_m1.shape) == 2 else result_m1
    m2_raw_bgr = cv2.cvtColor(binary_m2, cv2.COLOR_GRAY2BGR)
    m2_clean_bgr = cv2.cvtColor(cleaned_m2, cv2.COLOR_GRAY2BGR)

    # Top row: Original | Method 1 result
    top_row = np.hstack([original_resized, m1_bgr])
    # Bottom row: Method 2 raw binary | Method 2 cleaned
    bottom_row = np.hstack([m2_raw_bgr, m2_clean_bgr])

    # Add labels
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(top_row, "Original", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.putText(top_row, "Method 1: Kernel + Canny", (image.shape[1] + 10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(bottom_row, "Method 2: Otsu Binary", (10, 30), font, 0.7, (0, 255, 0), 2)
    cv2.putText(bottom_row, "Method 2: Cleaned", (image.shape[1] + 10, 30), font, 0.7, (0, 255, 0), 2)

    comparison = np.vstack([top_row, bottom_row])

    cv2.imwrite('comparison.jpg', comparison)
    print("\n[Comparison] Saved: comparison.jpg")

    # Show individual windows for closer inspection
    cv2.imshow('Original', image)
    cv2.imshow('Method 1 - Kernel + Canny', result_m1)
    cv2.imshow('Method 2 - Otsu Binary (raw)', binary_m2)
    cv2.imshow('Method 2 - Cleaned (clusters < 5 removed)', cleaned_m2)
    cv2.imshow('Comparison', comparison)
    print("Press any key to close all windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────
compare_methods(image)