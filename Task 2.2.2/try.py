import cv2
import numpy as np
from collections import deque


image = cv2.imread(r'C:\Users\parvc\Documents\College\ARK\Task 2.2.2\noisy.jpg')
if image is None:
    raise FileNotFoundError("Could not load 'noisy.jpg'. Make sure it is in the working directory.")


# ──────────────────────────────────────────────────────────────────────────────
# Custom edge detection — improved with multi-scale gradient + morphological
# cleanup to produce cleaner, more connected edges.
#
# Parameters (all tweakable at the bottom of the file):
#   edge_pre_blur    — Gaussian blur before edge detection to suppress noise
#   sobel_ksize      — Sobel kernel size (3, 5, or 7; larger = softer edges)
#   low_pct          — lower hysteresis percentile (0-100)
#   high_pct         — upper hysteresis percentile (0-100)
#   edge_dilate      — dilate edges by this many pixels to thicken them
#   edge_close_ksize — morphological closing kernel to bridge small gaps
# ──────────────────────────────────────────────────────────────────────────────
def custom_edge_detect(image_bgr, edge_pre_blur=11, sobel_ksize=7,
                       low_pct=60, high_pct=85, edge_dilate=1,
                       edge_close_ksize=3):
    # Pre-blur the image to suppress noise before computing gradients
    blurred = cv2.GaussianBlur(image_bgr, (edge_pre_blur, edge_pre_blur), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Multi-scale gradients: combine two Sobel scales for better coverage
    gx1 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_ksize)
    gy1 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_ksize)
    mag1 = np.sqrt(gx1**2 + gy1**2)

    # Second scale with smaller kernel for fine detail
    ksize2 = max(3, sobel_ksize - 2)
    gx2 = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=ksize2)
    gy2 = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=ksize2)
    mag2 = np.sqrt(gx2**2 + gy2**2)

    # Combine: take the max of both scales at each pixel
    magnitude = np.maximum(mag1, mag2)
    # Use direction from the scale with the stronger gradient
    direction = np.where(mag1 >= mag2, np.arctan2(gy1, gx1), np.arctan2(gy2, gx2))

    h, w = magnitude.shape
    suppressed = np.zeros_like(magnitude)
    angle = direction * 180.0 / np.pi
    angle[angle < 0] += 180

    # Non-maximum suppression
    for y in range(1, h - 1):
        for x in range(1, w - 1):
            a = angle[y, x]
            m = magnitude[y, x]
            if (0 <= a < 22.5) or (157.5 <= a <= 180):
                n1, n2 = magnitude[y, x - 1], magnitude[y, x + 1]
            elif 22.5 <= a < 67.5:
                n1, n2 = magnitude[y - 1, x + 1], magnitude[y + 1, x - 1]
            elif 67.5 <= a < 112.5:
                n1, n2 = magnitude[y - 1, x], magnitude[y + 1, x]
            else:
                n1, n2 = magnitude[y - 1, x - 1], magnitude[y + 1, x + 1]
            if m >= n1 and m >= n2:
                suppressed[y, x] = m

    nonzero = suppressed[suppressed > 0]
    if len(nonzero) == 0:
        return np.zeros((h, w), dtype=np.uint8)

    low_thresh = np.percentile(nonzero, low_pct)
    high_thresh = np.percentile(nonzero, high_pct)
    print(f"  Thresholds: low={low_thresh:.1f}, high={high_thresh:.1f}  "
          f"(range: {nonzero.min():.1f} - {nonzero.max():.1f})")

    strong = suppressed >= high_thresh
    weak = (suppressed >= low_thresh) & ~strong

    edges = np.zeros((h, w), dtype=np.uint8)
    edges[strong] = 255

    # Hysteresis
    prev_count = 0
    while True:
        dilated = cv2.dilate(edges, np.ones((3, 3), dtype=np.uint8))
        new_edges = (dilated > 0) & weak
        edges[new_edges] = 255
        curr_count = np.count_nonzero(edges)
        if curr_count == prev_count:
            break
        prev_count = curr_count

    # Morphological cleanup: close small gaps in edges
    if edge_close_ksize > 0:
        close_kern = np.ones((edge_close_ksize, edge_close_ksize), dtype=np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, close_kern)

    # Dilate to thicken edges (acts as wider barrier for the blur)
    if edge_dilate > 0:
        dilate_kern = np.ones((2 * edge_dilate + 1, 2 * edge_dilate + 1), dtype=np.uint8)
        edges = cv2.dilate(edges, dilate_kern)

    return edges


# ──────────────────────────────────────────────────────────────────────────────
# Edge-aware blur
#
# For each pixel, BFS flood-fills within a (kernel_size x kernel_size) window.
# The flood cannot cross or include edge pixels.  Only reachable non-edge
# pixels on the same side of the edge contribute to the average.
#
# Parameters:
#   img          — input BGR image (np.uint8)
#   edge_mask    — boolean array, True = edge pixel
#   kernel_size  — size of the blur window (must be odd)
#   iterations   — how many times to re-apply the blur for extra smoothing
# ──────────────────────────────────────────────────────────────────────────────
def edge_aware_blur(img, edge_mask, kernel_size=11, iterations=1):
    radius = kernel_size // 2
    h, w = img.shape[:2]

    current = img.copy()

    for it in range(iterations):
        print(f"  Pass {it + 1}/{iterations} (kernel={kernel_size}x{kernel_size})...")
        img_f = current.astype(np.float64)
        output = np.zeros_like(current, dtype=np.float64)

        # ── Step 1: blur all NON-edge pixels (BFS stays on same side) ──
        for y in range(h):
            if y % 40 == 0:
                print(f"    Row {y}/{h}")
            for x in range(w):
                # Skip edge pixels for now — handle them in step 2
                if edge_mask[y, x]:
                    continue

                # BFS from (y, x), staying inside the kernel window, blocked by edges
                visited = set()
                visited.add((y, x))
                queue = deque([(y, x)])
                total = img_f[y, x].copy()
                count = 1

                while queue:
                    cy, cx = queue.popleft()
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        ny, nx = cy + dy, cx + dx
                        if ny < 0 or ny >= h or nx < 0 or nx >= w:
                            continue
                        if abs(ny - y) > radius or abs(nx - x) > radius:
                            continue
                        if edge_mask[ny, nx]:
                            continue
                        if (ny, nx) in visited:
                            continue
                        visited.add((ny, nx))
                        queue.append((ny, nx))
                        total += img_f[ny, nx]
                        count += 1

                output[y, x] = total / count

        # ── Step 2: inpaint edge pixels from already-smoothed neighbours ──
        # Repeatedly fill edges from the outside in (handles thick dilated edges)
        filled = output.copy()
        edge_remaining = edge_mask.copy()
        max_passes = max(EDGE_DILATE * 2 + 2, 5)

        for p in range(max_passes):
            still_missing = False
            for y in range(h):
                for x in range(w):
                    if not edge_remaining[y, x]:
                        continue
                    total = np.zeros(3, dtype=np.float64)
                    count = 0
                    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1),
                                   (-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not edge_remaining[ny, nx]:
                            total += filled[ny, nx]
                            count += 1
                    if count > 0:
                        filled[y, x] = total / count
                        edge_remaining[y, x] = False
                    else:
                        still_missing = True
            if not still_missing:
                print(f"    Edge inpaint done in {p + 1} passes")
                break

        current = np.clip(filled, 0, 255).astype(np.uint8)

    return current


# ──────────────────────────────────────────────────────────────────────────────
# TWEAKABLE PARAMETERS — adjust these to tune the result
# ──────────────────────────────────────────────────────────────────────────────

# Edge detection parameters
EDGE_PRE_BLUR    = 15    # Gaussian blur before edge detection (odd, larger = less noise)
SOBEL_KSIZE      = 9    # Sobel kernel size: 3, 5, or 7 (larger = catches softer edges)
LOW_PCT          = 75    # Lower hysteresis percentile (lower = more edges)
HIGH_PCT         = 85    # Upper hysteresis percentile (lower = more strong edges)
EDGE_DILATE      = 1     # Thicken edges by N pixels (0 = no dilation)
EDGE_CLOSE_KSIZE = 3     # Morphological closing kernel to bridge gaps (0 = off)

# Blur parameters
KERNEL_SIZE      = 7     # Edge-aware blur window size (odd)
ITERATIONS       = 3     # Number of blur passes (more = smoother)

# ──────────────────────────────────────────────────────────────────────────────
# Run
# ──────────────────────────────────────────────────────────────────────────────
print("Detecting edges...")
edges = custom_edge_detect(image,
                           edge_pre_blur=EDGE_PRE_BLUR,
                           sobel_ksize=SOBEL_KSIZE,
                           low_pct=LOW_PCT,
                           high_pct=HIGH_PCT,
                           edge_dilate=EDGE_DILATE,
                           edge_close_ksize=EDGE_CLOSE_KSIZE)
edge_mask = edges > 0
print(f"  {np.count_nonzero(edges)} edge pixels detected")

print(f"\nRunning edge-aware blur: kernel={KERNEL_SIZE}, iterations={ITERATIONS}")
result = edge_aware_blur(image, edge_mask, kernel_size=KERNEL_SIZE, iterations=ITERATIONS)

cv2.imwrite('edge_aware_blur.jpg', result)
print(f"\n[Done] Saved -> edge_aware_blur.jpg")

# Display — including edges
cv2.imshow('Original Noisy', image)
cv2.imshow('Detected Edges', edges)
cv2.imshow('Edge-Aware Blur', result)
print("Press any key to close...")
cv2.waitKey(0)
cv2.destroyAllWindows()