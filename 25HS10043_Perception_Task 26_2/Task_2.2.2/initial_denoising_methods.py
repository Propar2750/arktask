import cv2
import numpy as np
import bm3d

# ──────────────────────────────────────────────────────────────────────────────
# Load the noisy image
# ──────────────────────────────────────────────────────────────────────────────
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
image = cv2.imread(os.path.join(SCRIPT_DIR, 'noisy.jpg'))
if image is None:
    raise FileNotFoundError("Could not load 'noisy.jpg'. Make sure it is in the working directory.")

print(f"Image loaded: {image.shape[1]}x{image.shape[0]}, {image.shape[2]} channels, dtype={image.dtype}")


# ──────────────────────────────────────────────────────────────────────────────
# Method A — Gaussian Blur
#   Simple low-pass filter; averages pixels using a Gaussian-weighted kernel.
#   Good baseline but blurs edges along with noise.
# ──────────────────────────────────────────────────────────────────────────────

gaussian_blurred = cv2.blur(image, (21, 21))
gaussian_blurred = cv2.GaussianBlur(gaussian_blurred, (7, 7), sigmaX=0)
gaussian_blurred = cv2.GaussianBlur(gaussian_blurred, (7, 7), sigmaX=0)
cv2.imwrite(os.path.join(SCRIPT_DIR, 'denoised_gaussian.jpg'), gaussian_blurred)
print("[A] Gaussian Blur (7x7) saved → denoised_gaussian.jpg")


# ──────────────────────────────────────────────────────────────────────────────
# Method B — Bilateral Filter
#   Smooths flat regions while preserving edges by weighting pixels based on
#   both spatial distance AND colour similarity.
#   d=9            — neighbourhood diameter
#   sigmaColor=75  — how different colours can be and still be mixed
#   sigmaSpace=75  — spatial extent of the kernel
# ──────────────────────────────────────────────────────────────────────────────
bilateral_filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
cv2.imwrite(os.path.join(SCRIPT_DIR, 'denoised_bilateral.jpg'), bilateral_filtered)
print("[B] Bilateral Filter saved → denoised_bilateral.jpg")


# ──────────────────────────────────────────────────────────────────────────────
# Method C — Non-Local Means Denoising (best for Gaussian noise)
#   Compares patches across the entire image and averages similar ones,
#   giving much better detail preservation than local filters.
#   h=10                  — filter strength for luminance
#   hForColorComponents=10 — filter strength for colour channels
#   templateWindowSize=7  — size of the patch used for comparison
#   searchWindowSize=21   — size of the area to search for similar patches
# ──────────────────────────────────────────────────────────────────────────────
nlmeans_denoised = cv2.fastNlMeansDenoisingColored(
    image, None, 30, 30, 5, 21
)
nlmeans_denoised = cv2.GaussianBlur(nlmeans_denoised, (3, 3), sigmaX=0)
cv2.imwrite(os.path.join(SCRIPT_DIR, 'denoised_nlmeans.jpg'), nlmeans_denoised)
print("[C] Non-Local Means Denoising saved → denoised_nlmeans.jpg")


# ──────────────────────────────────────────────────────────────────────────────
# Method D — Channel-wise denoising in YCrCb colour space
#
#   When Gaussian noise is added independently to each R, G, B channel, it
#   corrupts both luminance (brightness) and chrominance (colour) information.
#   By converting to YCrCb we separate these concerns:
#     Y  = luminance   → denoise lightly to preserve edges & detail
#     Cr = red chroma   → denoise aggressively (colour noise is very visible)
#     Cb = blue chroma  → denoise aggressively
#
#   Each channel is denoised independently with fastNlMeansDenoising (single-
#   channel version), then the channels are merged and converted back to BGR.
#   This gives much better results than treating all channels equally because
#   the human eye is far more sensitive to luminance detail than colour detail.
# ──────────────────────────────────────────────────────────────────────────────
ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
y, cr, cb = cv2.split(ycrcb)

# Denoise luminance lightly (h=6) to keep sharpness
y_denoised = cv2.fastNlMeansDenoising(y, None, 6, 7, 21)

# Denoise chrominance channels heavily (h=15) — colour noise is perceptually ugly
cr_denoised = cv2.fastNlMeansDenoising(cr, None, 15, 7, 21)
cb_denoised = cv2.fastNlMeansDenoising(cb, None, 15, 7, 21)

# Merge and convert back to BGR
ycrcb_denoised = cv2.merge([y_denoised, cr_denoised, cb_denoised])
channelwise_denoised = cv2.cvtColor(ycrcb_denoised, cv2.COLOR_YCrCb2BGR)

cv2.imwrite(os.path.join(SCRIPT_DIR, 'denoised_channelwise.jpg'), channelwise_denoised)
print("[D] Channel-wise YCrCb Denoising saved -> denoised_channelwise.jpg")


# ──────────────────────────────────────────────────────────────────────────────
# Method E — BM3D (Block-Matching and 3D Filtering)
#
#   State-of-the-art denoising algorithm that works in two stages:
#     1. Groups similar patches in the image by block-matching.
#     2. Applies collaborative filtering in a 3D transform domain.
#   This exploits non-local self-similarity (like NL-Means) but adds
#   sparsity-based shrinkage in the transform domain for much better results.
#
#   The noise analysis estimated sigma ~63 (on 0-255 scale).
#   BM3D expects the image normalised to [0, 1], so sigma_psd = 63/255 ≈ 0.25.
# ──────────────────────────────────────────────────────────────────────────────
print("[E] Running BM3D (this may take a moment)...")

# BM3D works on float images in [0, 1] range
image_float = image.astype(np.float64) / 255.0
sigma_psd = 63.0 / 255.0  # estimated noise sigma from examine.py

# Apply BM3D to each BGR channel independently
bm3d_channels = []
for i in range(3):
    denoised_ch = bm3d.bm3d(image_float[:, :, i], sigma_psd=sigma_psd,
                            stage_arg=bm3d.BM3DStages.ALL_STAGES)
    bm3d_channels.append(denoised_ch)

bm3d_denoised = np.stack(bm3d_channels, axis=2)
bm3d_denoised = np.clip(bm3d_denoised * 255, 0, 255).astype(np.uint8)

cv2.imwrite(os.path.join(SCRIPT_DIR, 'denoised_bm3d.jpg'), bm3d_denoised)
print("[E] BM3D Denoising saved -> denoised_bm3d.jpg")


# ──────────────────────────────────────────────────────────────────────────────
# Build side-by-side comparison grid
# ──────────────────────────────────────────────────────────────────────────────
font = cv2.FONT_HERSHEY_SIMPLEX

# Add labels to each image
def label(img, text):
    """Return a copy of img with a text label in the top-left corner."""
    out = img.copy()
    cv2.putText(out, text, (10, 30), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    return out

top_row = np.hstack([label(image, "Original"), label(gaussian_blurred, "A: Gaussian Blur")])
mid_row = np.hstack([label(bilateral_filtered, "B: Bilateral"), label(nlmeans_denoised, "C: NL-Means")])
bottom_row = np.hstack([label(channelwise_denoised, "D: YCrCb"), label(bm3d_denoised, "E: BM3D")])
comparison = np.vstack([top_row, mid_row, bottom_row])

cv2.imwrite(os.path.join(SCRIPT_DIR, 'comparison.jpg'), comparison)
print("[Comparison] Saved → comparison.jpg")


# ──────────────────────────────────────────────────────────────────────────────
# Display results
# ──────────────────────────────────────────────────────────────────────────────
cv2.imshow('Original', image)
cv2.imshow('A - Gaussian Blur', gaussian_blurred)
cv2.imshow('B - Bilateral Filter', bilateral_filtered)
cv2.imshow('C - Non-Local Means', nlmeans_denoised)
cv2.imshow('D - YCrCb Channel-wise', channelwise_denoised)
cv2.imshow('E - BM3D', bm3d_denoised)
cv2.imshow('Comparison', comparison)
print("Press any key to close all windows...")
cv2.waitKey(0)
cv2.destroyAllWindows()
