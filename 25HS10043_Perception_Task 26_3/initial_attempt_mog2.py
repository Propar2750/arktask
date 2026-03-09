"""
Task 2.3 — Medial Axis Detection of Moving Surgical Tools
==========================================================
Pipeline:
  1. Background Subtraction  (MOG2-based foreground mask)
  2. Image Cleaning           (morphological erosion + dilation)
  3. Edge & Line Detection    (Sobel derivatives + custom Hough Line Transform)
  4. Medial Axis Localization  (midpoint of opposite edge lines → overlay on frame)

Constraint: NO built-in cv2.HoughLines / cv2.HoughLinesP — Hough coded from scratch.
"""

import cv2
import numpy as np
import os
import math

# ── paths ────────────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR  = os.path.join(BASE_DIR, "input")    # 1.mp4, 2.mp4, 3.mp4 sit here
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
#  1. BACKGROUND SUBTRACTION
# ══════════════════════════════════════════════════════════════════════════════
def background_subtraction(frame, bg_subtractor):
    """
    Apply MOG2 background subtractor to isolate the moving foreground (tool).
    Returns a binary foreground mask (0 or 255).
    """
    fg_mask = bg_subtractor.apply(frame)
    # Threshold to get a clean binary mask (remove shadows marked as 127)
    _, fg_mask = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
    return fg_mask


# ══════════════════════════════════════════════════════════════════════════════
#  2. IMAGE CLEANING — morphological operations
# ══════════════════════════════════════════════════════════════════════════════
def clean_mask(fg_mask):
    """
    Use erosion and dilation (morphological open + close) to remove noise
    and fill holes, producing a clean silhouette of the tool.
    """
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))

    # Opening: erode then dilate — removes small noise blobs
    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    # Closing: dilate then erode — fills small holes inside the tool
    cleaned = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_large, iterations=3)

    # Extra dilation to ensure tool silhouette is solid
    cleaned = cv2.dilate(cleaned, kernel_small, iterations=1)

    return cleaned


# ══════════════════════════════════════════════════════════════════════════════
#  3a. EDGE DETECTION — Sobel derivatives
# ══════════════════════════════════════════════════════════════════════════════
def detect_edges(cleaned_mask):
    """
    Compute gradient magnitude using Sobel operators (first-order derivatives)
    and threshold to obtain edge pixels.
    """
    # Blur slightly to reduce minor noise before derivatives
    blurred = cv2.GaussianBlur(cleaned_mask, (5, 5), 1.0)

    # Sobel derivatives in X and Y
    grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Gradient magnitude
    magnitude = np.sqrt(grad_x ** 2 + grad_y ** 2)
    mag_max = magnitude.max()
    if mag_max == 0:
        magnitude = np.zeros_like(magnitude, dtype=np.uint8)
    else:
        magnitude = np.uint8(np.clip(magnitude / mag_max * 255, 0, 255))

    # Threshold to binary edges
    _, edges = cv2.threshold(magnitude, 50, 255, cv2.THRESH_BINARY)

    return edges


# ══════════════════════════════════════════════════════════════════════════════
#  3b. CUSTOM HOUGH LINE TRANSFORM  (from scratch — no cv2.HoughLines!)
# ══════════════════════════════════════════════════════════════════════════════
def custom_hough_lines(edge_image, rho_res=1, theta_res=np.pi / 180,
                       threshold_ratio=0.35):
    """
    Hough Line Transform implemented from scratch.

    Parameters
    ----------
    edge_image     : uint8 binary edge image (0/255)
    rho_res        : distance resolution in pixels (default 1 px)
    theta_res      : angle resolution in radians  (default 1°)
    threshold_ratio: fraction of max accumulator value used as vote threshold

    Returns
    -------
    lines : list of (rho, theta) detected lines
    """
    h, w = edge_image.shape[:2]

    # --- build parameter space ---------------------------------------------------
    diag = int(np.ceil(np.sqrt(h ** 2 + w ** 2)))         # max possible rho
    rhos   = np.arange(-diag, diag + 1, rho_res)
    thetas = np.arange(0, np.pi, theta_res)

    num_rhos   = len(rhos)
    num_thetas = len(thetas)

    # Precompute sin/cos lookup
    cos_t = np.cos(thetas)
    sin_t = np.sin(thetas)

    # --- accumulator -------------------------------------------------------------
    accumulator = np.zeros((num_rhos, num_thetas), dtype=np.int32)

    # Get edge pixel coordinates
    y_idxs, x_idxs = np.nonzero(edge_image)

    # Vote (vectorised over thetas for each edge pixel row-batch)
    # Process in batches for memory efficiency
    batch_size = 5000
    for start in range(0, len(y_idxs), batch_size):
        end = min(start + batch_size, len(y_idxs))
        xs = x_idxs[start:end].astype(np.float64)
        ys = y_idxs[start:end].astype(np.float64)

        # rho = x*cos(theta) + y*sin(theta)  — shape (batch, num_thetas)
        rho_vals = np.outer(xs, cos_t) + np.outer(ys, sin_t)
        rho_indices = np.round((rho_vals + diag) / rho_res).astype(np.int32)

        # Clip to valid range
        rho_indices = np.clip(rho_indices, 0, num_rhos - 1)

        for t_idx in range(num_thetas):
            rho_col = rho_indices[:, t_idx]
            # np.add.at handles repeated indices correctly
            np.add.at(accumulator[:, t_idx], rho_col, 1)

    # --- extract peaks -----------------------------------------------------------
    vote_threshold = int(threshold_ratio * accumulator.max()) if accumulator.max() > 0 else 1

    # Find all cells above threshold
    peak_rho_idxs, peak_theta_idxs = np.where(accumulator >= vote_threshold)

    # Sort by votes (strongest first)
    votes = accumulator[peak_rho_idxs, peak_theta_idxs]
    order = np.argsort(-votes)
    peak_rho_idxs   = peak_rho_idxs[order]
    peak_theta_idxs = peak_theta_idxs[order]

    # Non-maximum suppression: keep lines that are sufficiently apart
    lines = []
    rho_merge   = 30   # pixels
    theta_merge = np.deg2rad(10)

    for i in range(len(peak_rho_idxs)):
        rho_val   = rhos[peak_rho_idxs[i]]
        theta_val = thetas[peak_theta_idxs[i]]
        duplicate = False
        for (r, t) in lines:
            if abs(r - rho_val) < rho_merge and abs(t - theta_val) < theta_merge:
                duplicate = True
                break
        if not duplicate:
            lines.append((rho_val, theta_val))

    return lines


# ══════════════════════════════════════════════════════════════════════════════
#  Helper — draw a Hough line (rho, theta) on an image
# ══════════════════════════════════════════════════════════════════════════════
def draw_hough_line(image, rho, theta, color=(0, 255, 0), thickness=2):
    """Draw an infinite line parameterised by (rho, theta) on the image."""
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    length = max(image.shape[:2]) * 2
    pt1 = (int(x0 + length * (-b)), int(y0 + length * (a)))
    pt2 = (int(x0 - length * (-b)), int(y0 - length * (a)))
    cv2.line(image, pt1, pt2, color, thickness)


# ══════════════════════════════════════════════════════════════════════════════
#  Helper — classify lines into two groups (opposite edges of the tool)
# ══════════════════════════════════════════════════════════════════════════════
def classify_lines(lines):
    """
    Given Hough lines, group them into two clusters corresponding to the
    two long edges of the surgical tool.  We cluster by rho values for
    lines that have a similar theta (orientation).

    Returns two representative lines: (rho1, theta1), (rho2, theta2)
    or None if insufficient lines are found.
    """
    if len(lines) < 2:
        return None

    # Cluster by similar theta — the tool is basically one orientation.
    # Pick the dominant theta first.
    best_theta = lines[0][1]

    # Gather lines close in orientation to the dominant theta
    parallel_lines = []
    for (r, t) in lines:
        angle_diff = abs(t - best_theta)
        # Also check wrapping near 0 / pi
        angle_diff = min(angle_diff, np.pi - angle_diff)
        if angle_diff < np.deg2rad(20):
            parallel_lines.append((r, t))

    if len(parallel_lines) < 2:
        # Fall back: just use the two strongest lines
        return lines[0], lines[1]

    # Sort parallel lines by rho
    parallel_lines.sort(key=lambda x: x[0])

    # Take the two lines with the largest rho separation (two edges)
    line1 = parallel_lines[0]
    line2 = parallel_lines[-1]

    return line1, line2


# ══════════════════════════════════════════════════════════════════════════════
#  4. MEDIAL AXIS LOCALIZATION
# ══════════════════════════════════════════════════════════════════════════════
def compute_medial_axis_line(line1, line2):
    """
    Given two parallel-ish lines (rho1, theta1) and (rho2, theta2),
    compute the medial axis as the line midway between them.
    """
    rho1, theta1 = line1
    rho2, theta2 = line2
    mid_rho   = (rho1 + rho2) / 2.0
    mid_theta = (theta1 + theta2) / 2.0
    return (mid_rho, mid_theta)


def overlay_medial_axis(frame, medial_line, line1=None, line2=None,
                        draw_edges=True):
    """
    Draw the medial axis (and optionally the two edge lines) onto the frame.
    """
    overlay = frame.copy()

    if draw_edges and line1 is not None and line2 is not None:
        draw_hough_line(overlay, line1[0], line1[1], color=(0, 255, 0), thickness=1)
        draw_hough_line(overlay, line2[0], line2[1], color=(0, 255, 0), thickness=1)

    # Medial axis in red, thicker
    draw_hough_line(overlay, medial_line[0], medial_line[1],
                    color=(0, 0, 255), thickness=2)

    return overlay


# ══════════════════════════════════════════════════════════════════════════════
#  FULL PIPELINE
# ══════════════════════════════════════════════════════════════════════════════
def process_video(video_path, output_video_path=None, save_debug_frames=False):
    """
    End-to-end pipeline:
      frame → bg subtraction → morphological cleaning → edge detection
      → custom Hough lines → medial axis → overlay on original frame
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps    = cap.get(cv2.CAP_PROP_FPS) or 30
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"  Resolution : {width}x{height}")
    print(f"  FPS        : {fps:.1f}")
    print(f"  Frames     : {total}")

    # Output video writer
    if output_video_path is None:
        name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(OUTPUT_DIR, f"{name}_medial_axis.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out    = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    # Debug output directory
    debug_dir = None
    if save_debug_frames:
        name = os.path.splitext(os.path.basename(video_path))[0]
        debug_dir = os.path.join(OUTPUT_DIR, f"{name}_debug")
        os.makedirs(debug_dir, exist_ok=True)

    # Background subtractor (MOG2)
    bg_sub = cv2.createBackgroundSubtractorMOG2(
        history=200, varThreshold=50, detectShadows=True
    )

    frame_idx   = 0
    prev_medial = None   # remember last good medial axis for continuity

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # ── Stage 1: Background Subtraction ──────────────────────────────
        fg_mask = background_subtraction(frame, bg_sub)

        # ── Stage 2: Image Cleaning ──────────────────────────────────────
        cleaned = clean_mask(fg_mask)

        # ── Stage 3a: Edge Detection (Sobel) ─────────────────────────────
        edges = detect_edges(cleaned)

        # ── Stage 3b: Custom Hough Line Transform ────────────────────────
        lines = custom_hough_lines(edges, rho_res=1,
                                   theta_res=np.pi / 180,
                                   threshold_ratio=0.30)

        # ── Stage 4: Medial Axis ─────────────────────────────────────────
        overlay = frame.copy()
        pair = classify_lines(lines) if len(lines) >= 2 else None

        if pair is not None:
            line1, line2 = pair
            medial = compute_medial_axis_line(line1, line2)
            prev_medial = (line1, line2, medial)
            overlay = overlay_medial_axis(frame, medial, line1, line2,
                                          draw_edges=True)
        elif prev_medial is not None:
            # Use previous detection for continuity
            line1, line2, medial = prev_medial
            overlay = overlay_medial_axis(frame, medial, line1, line2,
                                          draw_edges=True)

        out.write(overlay)

        # Save debug frames for a few selected frames
        if save_debug_frames and frame_idx % max(1, total // 10) == 0:
            cv2.imwrite(os.path.join(debug_dir, f"f{frame_idx:04d}_1_fg.png"),
                        fg_mask)
            cv2.imwrite(os.path.join(debug_dir, f"f{frame_idx:04d}_2_clean.png"),
                        cleaned)
            cv2.imwrite(os.path.join(debug_dir, f"f{frame_idx:04d}_3_edges.png"),
                        edges)
            cv2.imwrite(os.path.join(debug_dir, f"f{frame_idx:04d}_4_overlay.png"),
                        overlay)

        if frame_idx % 50 == 0 or frame_idx == 1:
            n_lines = len(lines) if lines else 0
            print(f"    Frame {frame_idx:>5}/{total}  |  "
                  f"Hough lines: {n_lines:>3}  |  "
                  f"Pair found: {'Yes' if pair else 'No'}")

    cap.release()
    out.release()
    print(f"  ✓ Saved → {output_video_path}\n")
    return output_video_path


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    videos = ["1.mp4", "2.mp4", "3.mp4"]

    for v in videos:
        path = os.path.join(VIDEO_DIR, v)
        if not os.path.isfile(path):
            print(f"[!] Video not found: {path}, skipping.")
            continue
        print(f"\n{'='*60}")
        print(f"Processing: {v}")
        print(f"{'='*60}")
        process_video(path, save_debug_frames=True)

    print("\n All videos processed. Output in:", OUTPUT_DIR)
