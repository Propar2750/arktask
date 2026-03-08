# Task 2.3 — Medial Axis Detection of Moving Surgical Tools

**Author:** [Your Name], [Other Contributors]

---

## Abstract

This report presents a computer vision pipeline for detecting the medial axis (centreline) of moving surgical tools in video. The pipeline consists of background subtraction, morphological cleaning, Sobel-based edge detection, and a fully custom Hough Line Transform (no `cv2.HoughLines` or `cv2.HoughLinesP`). Two background subtraction approaches were implemented: per-pixel median and MOG2 (Mixture of Gaussians). Detected edge lines are classified by orientation, and the medial axis is computed as the midline between the two widest-separated parallel edge lines. The output is an annotated video with green edges and a red medial axis overlaid on each frame.

[TODO: Add any context about the surgical tool tracking application and the input videos.]

---

## I. Introduction

The objective of Task 2.3 is to detect the medial axis (centreline) of a moving surgical tool in video footage. The medial axis is the line equidistant from both edges of the tool — it represents the tool's orientation and position.

Key constraints:
- **No built-in Hough Transform**: `cv2.HoughLines` and `cv2.HoughLinesP` are not allowed. The Hough Line Transform must be implemented from scratch.
- The pipeline must work on video (frame-by-frame processing).
- Output: overlay of detected edges (green) and medial axis (red) on each video frame.

The pipeline processes three videos (`1.mp4`, `2.mp4`, `3.mp4`) and produces annotated output videos.

---

## II. Problem Statement

Given video footage of a moving surgical tool:
1. **Segment** the tool from the background.
2. **Detect** the edges of the tool.
3. **Find** the dominant edge lines using a custom Hough Transform.
4. **Compute** the medial axis as the midline between opposite edges.
5. **Overlay** edges and medial axis on each frame and produce output video.

Constraint: The Hough Line Transform must be implemented from scratch — no OpenCV Hough functions.

[TODO: Add sample frames from the input videos.]

---

## III. Related Work

- **Background subtraction**: Separating foreground (moving objects) from a static background. Methods range from simple frame differencing to statistical models (MOG, MOG2, KNN).
- **Hough Line Transform** [1]: Maps edge points to (ρ, θ) parameter space via voting. Peaks in the accumulator correspond to dominant lines.
- **Medial axis / skeleton**: The set of points equidistant from the boundary. For elongated objects, it reduces to the centreline. Can be computed via morphological thinning or from boundary lines.
- **Sobel edge detection**: Gradient-based edge detection using directional derivative kernels.

[1] R. O. Duda and P. E. Hart, "Use of the Hough Transformation to Detect Lines and Curves in Pictures," Communications of the ACM, Vol. 15, No. 1, pp. 11–15, 1972.

---

## IV. Initial Attempts

### Approach 1 — Median Background + Custom Hough (`me.py`)

**Background Subtraction:**
- Compute the per-pixel median across ALL frames of the video → static background model.
- For each frame, compute `|frame - background|` per pixel. Pixels exceeding a threshold (30) are marked as foreground.

**Strengths:** Simple, deterministic, no training period.
**Weaknesses:** Requires loading all frames into memory. Cannot adapt to gradual background changes.

**Edge Detection:**
- Sobel derivatives (X and Y) on the foreground mask → gradient magnitude → threshold to get binary edges.

**Custom Hough Transform:**
- For each edge pixel (x, y), vote in the (ρ, θ) accumulator:

$$\rho = x \cos\theta + y \sin\theta$$

  for θ from 0° to 179° in 1° steps.
- **Vectorized voting**: batch all edge pixels for each θ value (NumPy vectorization instead of per-pixel loops).
- **Peak extraction**: find accumulator cells above a threshold.
- **Non-maximum suppression**: merge nearby peaks (similar ρ and θ) to avoid duplicate detections.

**Medial Axis:**
- Identify opposite edges as the pair of detected lines with the same orientation (similar θ) but maximum ρ separation.
- Medial axis = line with averaged ρ and same θ.
- Uses previous frame's detection for temporal continuity when current detection fails.

---

## V. Final Approach

### Enhanced Pipeline — MOG2 Background + Improved Hough (`medial_axis.py`)

#### Step 1 — Background Subtraction (MOG2)

```python
bg_subtractor = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=True
)
```

MOG2 models each pixel's history as a mixture of Gaussians. Advantages over median:
- **Adaptive**: continuously updates the background model.
- **Shadow detection**: marks shadow pixels (127) separately from foreground (255).
- **No memory overhead**: doesn't need to store all frames.

Shadows are removed by thresholding the mask at 200.

#### Step 2 — Morphological Cleaning

```python
# Open (erosion→dilation) to remove small noise
cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
# Close (dilation→erosion) to fill holes
cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large)
# Extra dilation to ensure tool boundary is solid
cv2.dilate(mask, kernel_small)
```

#### Step 3 — Edge Detection (Sobel)

```python
blurred = cv2.GaussianBlur(mask, (5, 5), 0)
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
magnitude = sqrt(sobel_x² + sobel_y²)
edges = (magnitude > threshold)
```

#### Step 4 — Custom Hough Line Transform

The Hough transform is implemented entirely from scratch:

**Accumulator setup:**
- θ range: 0° to 179° (1° steps → 180 bins)
- ρ range: -diagonal to +diagonal (1-pixel steps)
- Accumulator: 2D array of vote counts

**Voting (vectorized):**
```python
for t_idx in range(num_thetas):
    rhos = edge_x * cos(theta[t_idx]) + edge_y * sin(theta[t_idx])
    rho_indices = (rhos - rho_min).round().astype(int)
    # Vote using np.add.at for safe accumulation
    np.add.at(accumulator, (rho_indices, t_idx), 1)
```

**Peak finding:**
- Threshold: accumulator cells with votes ≥ threshold
- Non-maximum suppression: for each group of nearby peaks (|Δρ| < merge_dist and |Δθ| < merge_angle), keep only the strongest.

#### Step 5 — Line Classification & Medial Axis

1. **Cluster by orientation**: group detected lines by θ (within ±5°).
2. **Pick dominant orientation**: the cluster with the most lines.
3. **Find edge pair**: within the dominant cluster, find the two lines with the widest ρ separation — these are the tool's opposite edges.
4. **Medial axis**: $\rho_{mid} = (\rho_1 + \rho_2) / 2$, same θ.

#### Step 6 — Visualization

- **Green** lines: detected edges.
- **Red** line: medial axis (centreline).
- Output saved as MP4 video.
- Debug frames saved at selected intervals showing: FG mask, cleaned mask, edges, and overlay.

### Key Equations

Hough voting:
$$\rho = x \cos\theta + y \sin\theta$$

Medial axis from edge pair $(\rho_1, \theta)$ and $(\rho_2, \theta)$:
$$\rho_{med} = \frac{\rho_1 + \rho_2}{2}, \quad \theta_{med} = \theta$$

Line rendering from (ρ, θ):
$$x = \rho \cos\theta - t \sin\theta, \quad y = \rho \sin\theta + t \cos\theta$$
where $t$ ranges across the image.

---

## VI. Results and Observation

[TODO: Add screenshots of the output for each video (1.mp4, 2.mp4, 3.mp4). Include:]
- A frame showing the foreground mask
- A frame showing detected edges (green overlay)
- A frame showing the medial axis (red overlay)

[TODO: Compare the two approaches:]

| Aspect | Median Background (`me.py`) | MOG2 (`medial_axis.py`) |
|--------|----------------------------|------------------------|
| Memory Usage | High (all frames loaded) | Low (streaming) |
| Adaptability | None (static model) | Continuous adaptation |
| Shadow Handling | None | MOG2 shadow detection |
| Speed | Fast per-frame (after setup) | Fast per-frame |
| Accuracy | [TODO] | [TODO] |

[TODO: Discuss any failure cases — when did the Hough transform miss lines? When did the medial axis jump?]

---

## VII. Future Work

- **Temporal smoothing**: Apply Kalman filtering or exponential moving average to the medial axis parameters (ρ, θ) across frames for smoother tracking.
- **Multi-tool tracking**: Extend to detect and track multiple tools simultaneously by clustering Hough peaks into separate tool groups.
- **Deep learning segmentation**: Replace the background subtraction + edge detection pipeline with a trained segmentation model (U-Net, Mask R-CNN) for more robust tool detection.
- **Sub-pixel Hough**: Interpolate in the accumulator space for sub-degree angular resolution.
- **Real-time optimization**: The current frame-by-frame processing could be parallelized or GPU-accelerated for real-time surgical assistance applications.

---

## Conclusion

A complete pipeline for medial axis detection of moving surgical tools was implemented, featuring a fully custom Hough Line Transform (no built-in OpenCV Hough functions). Two background subtraction methods were compared, with MOG2 providing better adaptability. The custom Hough transform uses vectorized NumPy operations for efficiency and non-maximum suppression for clean line detection. The medial axis is computed as the midline between the tool's opposite edges, producing annotated video output.

[TODO: Add final observations about difficulties faced and lessons learned.]

---

## References

[1] R. O. Duda and P. E. Hart, "Use of the Hough Transformation to Detect Lines and Curves in Pictures," Communications of the ACM, Vol. 15, No. 1, pp. 11–15, 1972.

[2] Z. Zivkovic, "Improved Adaptive Gaussian Mixture Model for Background Subtraction," Proc. ICPR, Vol. 2, pp. 28–31, 2004.

[3] [TODO: Add any references to surgical tool tracking papers or resources.]

---

*Acknowledgements: [TODO: Add anyone who helped.]*
