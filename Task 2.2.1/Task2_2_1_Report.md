# Task 2.2.1 — Image Cleaning and Line Prediction

**Author:** [Your Name], [Other Contributors]

---

## Abstract

This report addresses two image processing sub-tasks: (1) cleaning a noisy binary image of Iron Man using custom kernel filtering and connected-component analysis, and (2) predicting and completing broken lines in the cleaned image using Hough Line Transform and morphological operations. Two cleaning methods were implemented — a custom 5×5 weighted-average kernel with Canny edge removal, and Otsu's binarization with small-cluster pruning. Line prediction extends detected line segments to image boundaries and fills gaps using dilation and morphological closing.

[TODO: Add details about the source image and any specific requirements from the task description.]

---

## I. Introduction

The goal of Task 2.2.1 is to process a noisy image (`iron_man_noisy.jpg`) through two stages:
1. **Image Cleaning**: Remove noise while preserving the underlying structure (lines/shapes).
2. **Line Prediction**: Detect line segments in the cleaned image, extend them, and fill gaps to produce a complete line drawing.

The challenge lies in balancing noise removal with structure preservation — aggressive filtering removes noise but also destroys thin lines, while gentle filtering leaves residual noise that confuses line detection.

---

## II. Problem Statement

Given a noisy grayscale image (`iron_man_noisy.jpg`):
1. **Clean** the image to produce a clear binary representation of the underlying lines/shapes.
2. **Detect** line segments in the cleaned image.
3. **Extend** detected lines to the image boundaries.
4. **Fill gaps** between broken line segments to produce a complete, continuous line drawing.

[TODO: Add the original noisy image and any specific constraints mentioned in the task.]

---

## III. Related Work

- **Gaussian/Median filtering**: Standard approaches for noise removal. Gaussian filtering smooths uniformly; median filtering preserves edges better but can remove thin structures.
- **Otsu's thresholding**: Automatic binary thresholding that minimizes intra-class variance. Effective when the histogram is bimodal.
- **Connected-component analysis**: Labels contiguous regions; useful for removing small noise clusters by area.
- **Hough Line Transform**: Maps edge points to parameter space (ρ, θ) to detect lines. The probabilistic variant (HoughLinesP) detects line segments directly.
- **Morphological operations**: Dilation, erosion, opening, and closing are foundational tools for cleaning binary images and connecting broken structures.

[TODO: Add any specific references.]

---

## IV. Initial Attempts

### Method 1 — Custom Kernel Filtering + Canny Edge Removal (`kernel_filter.py`)

**Steps:**
1. **Custom 5×5 weighted-average kernel**: A hand-designed kernel emphasizing neighbouring pixels while de-weighting the centre pixel (centre weight = 0):

$$K = \frac{1}{25} \begin{bmatrix} 0 & 1 & 1 & 1 & 0 \\ 1 & 1.25 & 2 & 1.25 & 1 \\ 1 & 2 & 0 & 2 & 1 \\ 1 & 1.25 & 2 & 1.25 & 1 \\ 0 & 1 & 1 & 1 & 0 \end{bmatrix}$$

   The zero centre weight means the output at each pixel depends entirely on its neighbours, providing aggressive smoothing.

2. **Binary thresholding** at intensity 90 — pixels above 90 become white (255), below become black (0).

3. **Canny edge detection** on the binary image, with the detected edges removed (set to 0). This cleans up boundary artifacts from the thresholding step.

**Results:** Produces a clean binary image but may lose thin line details due to the aggressive smoothing kernel.

### Method 2 — Otsu Binarization + Cluster Removal (`kernel_filter.py`)

**Steps:**
1. **Otsu's automatic thresholding**: Computes the optimal threshold to separate foreground (lines) from background (noise + empty space). No manual threshold tuning required.

2. **Connected-component analysis** (8-connectivity): Labels every contiguous region.

3. **Small cluster removal**: Any connected component with fewer than 7 pixels is removed (set to 0). This eliminates isolated noise speckles while preserving genuine line segments (which are larger).

**Results:** Better structure preservation than Method 1, since Otsu picks an optimal threshold and cluster removal targets noise specifically.

[TODO: Add comparison images showing Method 1 vs Method 2 outputs.]

---

## V. Final Approach

### Line Prediction and Completion (`line_prediction.py`)

Starting from the cleaned binary image, the line prediction pipeline:

1. **Edge detection**: Canny edge detector (thresholds 50, 150) extracts edge pixels from the binary image.

2. **Hough Line Transform**: `cv2.HoughLinesP` with parameters:
   - ρ resolution: 1 pixel
   - θ resolution: π/180 radians
   - Threshold: 50 votes
   - Minimum line length: 30 pixels
   - Maximum line gap: 10 pixels

3. **Line extension**: For each detected segment, compute the slope and y-intercept, then extend the line to the image boundaries:
   - For non-vertical lines: compute intersections with x=0 and x=width-1, clamp y values.
   - For vertical lines: draw from y=0 to y=height-1.

4. **Gap filling**: Morphological operations on the original binary image:
   - **Dilation** (5×5 rectangular kernel, 2 iterations): expands line segments to connect nearby fragments.
   - **Morphological closing** (5×5 kernel, 2 iterations): dilation followed by erosion — bridges gaps without changing overall thickness.

5. **Line thinning**: Uses `cv2.ximgproc.thinning()` (Zhang-Suen algorithm) if available, otherwise falls back to erosion with a 3×3 elliptical kernel. Produces single-pixel-wide lines.

### Outputs

- `iron_man_lines_detected.jpg` — detected and extended lines overlaid on the image
- `iron_man_lines_filled.jpg` — gaps filled via morphological closing
- `iron_man_lines_thinned.jpg` — thinned single-pixel lines

[TODO: Add the output images here.]

---

## VI. Results and Observation

[TODO: Add comparison images and observations. Consider:]
- How well Method 1 vs Method 2 cleaned the noisy image
- Whether the Hough Transform detected all major lines
- How effective the gap-filling was — were there still missing segments?
- Quality of the thinned output

[TODO: Add a comparison table if applicable.]

---

## VII. Future Work

- **Adaptive kernel design**: Learn the optimal kernel weights from training data rather than hand-designing them.
- **Deep learning denoising**: A trained denoising autoencoder could outperform hand-crafted filters on complex noise patterns.
- **Iterative Hough + fill loop**: Alternate between line detection and gap filling — newly filled segments may reveal additional lines in the next iteration.
- **Spline-based gap filling**: Instead of morphological dilation, fit smooth curves between nearby line endpoints for more natural gap completion.

---

## Conclusion

Two image cleaning methods were implemented and compared for removing noise from a binary image. Otsu's binarization with connected-component cluster removal (Method 2) provided better structure preservation than custom kernel filtering (Method 1). A line prediction pipeline using Hough Line Transform, morphological gap filling, and thinning successfully extended and completed broken lines in the cleaned image.

[TODO: Add final observations and lessons learned.]

---

## References

[1] N. Otsu, "A Threshold Selection Method from Gray-Level Histograms," IEEE Trans. Systems, Man, and Cybernetics, Vol. 9, No. 1, pp. 62–66, 1979.

[2] J. Canny, "A Computational Approach to Edge Detection," IEEE Trans. Pattern Analysis and Machine Intelligence, Vol. 8, No. 6, pp. 679–698, 1986.

[3] R. O. Duda and P. E. Hart, "Use of the Hough Transformation to Detect Lines and Curves in Pictures," Communications of the ACM, Vol. 15, No. 1, pp. 11–15, 1972.

[4] [TODO: Add any other references.]

---

*Acknowledgements: [TODO: Add anyone who helped.]*
