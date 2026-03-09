import cv2
import numpy as np
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ──────────────────────────────────────────────────────────────────────────────
# Load image
# ──────────────────────────────────────────────────────────────────────────────
image = cv2.imread(os.path.join(SCRIPT_DIR, 'iron_man_noisy.jpg'))
if image is None:
    raise FileNotFoundError("Could not load 'iron_man_noisy.jpg'. Make sure it is in the working directory.")


# ──────────────────────────────────────────────────────────────────────────────
# Final Approach – Otsu binarisation + connected-component cluster removal
#
# Steps:
#   1. Convert to grayscale and apply Otsu's automatic threshold
#      to produce a clean binary image.
#   2. Find all connected components (8-connectivity).
#   3. Remove any cluster (connected region) with fewer than 7 pixels.
# ──────────────────────────────────────────────────────────────────────────────
def clean_image(image, min_cluster_size=7):
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
    cv2.imwrite(os.path.join(SCRIPT_DIR, 'method2_cleaned.jpg'), cleaned)
    print(f"Removed {removed_count} clusters with < {min_cluster_size} pixels.")
    print("Saved: method2_cleaned.jpg")

    return cleaned


# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
result = clean_image(image)

cv2.imshow('Cleaned Image', result)
print("\nPress any key to close the window...")
cv2.waitKey(0)
cv2.destroyAllWindows()
