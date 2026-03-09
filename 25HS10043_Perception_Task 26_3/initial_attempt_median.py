"""
Task 2.3 — Medial Axis Detection of Moving Surgical Tools
==========================================================
Pipeline:
  1. Background Subtraction   (median-based static background model)
  2. Image Cleaning           (morphological erosion + dilation)
  3. Edge & Line Detection    (Sobel derivatives + custom Hough Line Transform)
  4. Medial Axis Localization  (midpoint of opposite edge lines → overlay on frame)

Constraint: NO built-in cv2.HoughLines / cv2.HoughLinesP — Hough coded from scratch.
"""

import cv2
import numpy as np
import os

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR  = os.path.join(BASE_DIR, "input")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. BACKGROUND SUBTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def get_background_image(video_path):
    """
    Build a static background model by computing the per-pixel median
    across every frame of the video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    background = np.median(np.array(frames), axis=0).astype(np.uint8)
    return background


def subtract_background(frame, background, threshold=30):
    """
    Compare each pixel to the background; pixels whose absolute difference
    exceeds `threshold` are marked as foreground (255), rest as background (0).
    """
    diff      = cv2.absdiff(frame, background)
    gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray_diff, threshold, 255, cv2.THRESH_BINARY)
    return fg_mask


# ══════════════════════════════════════════════════════════════════════════════
#  2. IMAGE CLEANING  (morphological operations)
# ══════════════════════════════════════════════════════════════════════════════

def clean_mask(fg_mask):
    """
    Remove noise and fill holes using morphological opening and closing.
      • Opening (erode → dilate): removes small noise blobs
      • Closing (dilate → erode): fills small holes inside the tool
    """
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel_small, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_large, iterations=3)
    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
#  3a. EDGE DETECTION  (Sobel derivatives)
# ══════════════════════════════════════════════════════════════════════════════

def detect_edges(mask):
    """
    Compute gradient magnitude via Sobel first-order derivatives (dx, dy)
    and threshold to obtain a binary edge map.
    """
    blurred = cv2.GaussianBlur(mask, (5, 5), 1.0)

    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    mag_max   = magnitude.max()
    if mag_max == 0:
        return np.zeros_like(mask)

    magnitude = np.uint8(magnitude / mag_max * 255)
    _, edges  = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)
    return edges


# ══════════════════════════════════════════════════════════════════════════════
#  3b. CUSTOM HOUGH LINE TRANSFORM  (from scratch — NO cv2.HoughLines)
# ══════════════════════════════════════════════════════════════════════════════
#
#  Each edge pixel (x, y) votes for every line ρ = x·cos θ + y·sin θ
#  that passes through it.  The accumulator stores vote counts in (ρ, θ) space.
#

def hough_transform(edges, rho_res=1, theta_res_deg=1):
    """
    Vectorised Hough Line Transform.

    Returns
    -------
    accumulator : 2-D vote array  (num_rhos × num_thetas)
    rhos        : 1-D array of ρ bin centres
    thetas      : 1-D array of θ values in radians
    """
    h, w = edges.shape
    diag  = int(np.ceil(np.sqrt(h**2 + w**2)))

    rhos   = np.arange(-diag, diag + 1, rho_res)
    thetas = np.deg2rad(np.arange(0, 180, theta_res_deg))

    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    accumulator = np.zeros((len(rhos), len(thetas)), dtype=np.int32)

    ys, xs = np.nonzero(edges)                        # edge pixel coords

    # Vote in batches to keep memory reasonable
    BATCH = 5000
    for lo in range(0, len(ys), BATCH):
        hi   = min(lo + BATCH, len(ys))
        bx   = xs[lo:hi].astype(np.float64)
        by   = ys[lo:hi].astype(np.float64)

        # ρ = x·cosθ + y·sinθ  →  shape (batch, num_thetas)
        rho_vals = np.outer(bx, cos_t) + np.outer(by, sin_t)
        rho_idxs = np.round((rho_vals + diag) / rho_res).astype(np.int32)
        rho_idxs = np.clip(rho_idxs, 0, len(rhos) - 1)

        for t in range(len(thetas)):
            np.add.at(accumulator[:, t], rho_idxs[:, t], 1)

    return accumulator, rhos, thetas


def find_lines(accumulator, rhos, thetas, threshold_ratio=0.35,
               rho_merge=30, theta_merge_deg=10):
    """
    Extract the strongest lines from the accumulator with non-maximum
    suppression so that nearby duplicates are merged.

    Returns list of (rho, theta) tuples sorted by vote strength.
    """
    vote_thresh = int(threshold_ratio * accumulator.max()) if accumulator.max() > 0 else 1
    ri, ti = np.where(accumulator >= vote_thresh)

    votes = accumulator[ri, ti]
    order = np.argsort(-votes)
    ri, ti = ri[order], ti[order]

    theta_merge = np.deg2rad(theta_merge_deg)
    lines = []
    for k in range(len(ri)):
        r, t = rhos[ri[k]], thetas[ti[k]]
        # Skip if too close to an already-accepted line
        if any(abs(r - lr) < rho_merge and abs(t - lt) < theta_merge
               for lr, lt in lines):
            continue
        lines.append((r, t))

    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  4. MEDIAL AXIS LOCALIZATION
# ══════════════════════════════════════════════════════════════════════════════

def classify_edge_pair(lines):
    """
    Among all detected lines, pick the two parallel edges of the tool
    (largest ρ separation at similar θ).  Returns (line1, line2) or None.
    """
    if len(lines) < 2:
        return None

    best_theta = lines[0][1]
    parallel = [(r, t) for r, t in lines
                if min(abs(t - best_theta), np.pi - abs(t - best_theta))
                < np.deg2rad(20)]

    if len(parallel) < 2:
        return lines[0], lines[1]          # fallback: two strongest

    parallel.sort(key=lambda x: x[0])
    return parallel[0], parallel[-1]       # widest ρ gap = opposite edges


def medial_axis_line(line1, line2):
    """Average two (ρ, θ) lines to get the midline (medial axis)."""
    return ((line1[0] + line2[0]) / 2.0,
            (line1[1] + line2[1]) / 2.0)


# ── drawing helpers ──────────────────────────────────────────────────────────

def draw_line(image, rho, theta, color, thickness=2):
    """Draw an infinite (ρ, θ) line across the image."""
    a, b = np.cos(theta), np.sin(theta)
    x0, y0 = a * rho, b * rho
    L = max(image.shape[:2]) * 2
    pt1 = (int(x0 - L * b), int(y0 + L * a))
    pt2 = (int(x0 + L * b), int(y0 - L * a))
    cv2.line(image, pt1, pt2, color, thickness)


def overlay_result(frame, line1, line2, medial):
    """Draw edge lines (green) and medial axis (red) on the frame."""
    vis = frame.copy()
    draw_line(vis, line1[0], line1[1], color=(0, 255, 0), thickness=1)
    draw_line(vis, line2[0], line2[1], color=(0, 255, 0), thickness=1)
    draw_line(vis, medial[0], medial[1], color=(0, 0, 255), thickness=2)
    return vis


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def process_video(video_path, idx):
    """Run the complete 4-stage pipeline on one video."""
    print(f"\n{'='*60}")
    print(f"Processing: {os.path.basename(video_path)}")
    print(f"{'='*60}")

    # ── Stage 1 prep: build background model ──
    print("  Building background model (median of all frames)…")
    background = get_background_image(video_path)
    bg_path = os.path.join(OUTPUT_DIR, f"background_{idx}.jpg")
    cv2.imwrite(bg_path, background)

    # ── Open video for frame-by-frame processing ──
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"  {w}x{h} @ {fps:.0f} fps, {total} frames")

    out_path = os.path.join(OUTPUT_DIR, f"medial_axis_{idx}.mp4")
    fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
    out      = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    prev_result = None          # last good (line1, line2, medial) for continuity
    frame_num   = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        # Stage 1 — Background subtraction
        fg_mask = subtract_background(frame, background)

        # Stage 2 — Morphological cleaning
        cleaned = clean_mask(fg_mask)

        # Stage 3a — Edge detection (Sobel derivatives)
        edges = detect_edges(cleaned)

        # Stage 3b — Custom Hough Line Transform
        acc, rhos, thetas = hough_transform(edges)
        lines = find_lines(acc, rhos, thetas)

        # Stage 4 — Medial axis
        pair = classify_edge_pair(lines)
        if pair is not None:
            l1, l2  = pair
            medial  = medial_axis_line(l1, l2)
            prev_result = (l1, l2, medial)
            result  = overlay_result(frame, l1, l2, medial)
        elif prev_result is not None:
            result = overlay_result(frame, *prev_result)
        else:
            result = frame

        out.write(result)

        if frame_num % 50 == 0 or frame_num == 1:
            print(f"    Frame {frame_num:>5}/{total}  |  "
                  f"Lines: {len(lines):>3}  |  "
                  f"Pair: {'Yes' if pair else 'No'}")

    cap.release()
    out.release()
    print(f"  ✓ Saved → {out_path}")


# ── main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for i in range(1, 4):
        path = os.path.join(VIDEO_DIR, f"{i}.mp4")
        if not os.path.isfile(path):
            print(f"[!] Not found: {path}, skipping.")
            continue
        process_video(path, i)

    print(f"\nAll done. Output in: {OUTPUT_DIR}")
