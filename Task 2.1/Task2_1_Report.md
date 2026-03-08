# Task 2.1 — Drone Line Follower

**Author:** [Your Name], [Other Contributors]

---

## Abstract

This report presents an image-based line-following algorithm for a Parrot Mambo minidrone. The drone's downward-facing camera provides 120×160 RGB frames; the algorithm detects a red line and outputs a unit steering vector (X, Y) to guide the drone along the line. Three progressively refined approaches were developed: a 7×7 grid-based scanner, a 3-strip tangent method, and a global+perpendicular centroid method with heading-aware clamping. The final algorithm handles turns up to 150°+ with no directional bias, operates entirely with fixed-size arrays for Simulink compatibility, and maintains a persistent heading to prevent oscillation.

[TODO: Add any additional context about the testing environment, drone hardware, or competition details.]

---

## I. Introduction

The objective of Task 2.1 is to steer a drone so that it follows a red line painted on the ground. The input to the algorithm is three 120×160 `uint8` arrays (R, G, B channels from the downward camera), and the output is a unit vector (X, Y) where X points upward in the image (forward in the drone's frame) and Y points rightward.

Key challenges include:
- **Sharp turns** up to 150° where the line bends back almost the way the drone came.
- **Simulink compatibility**: the algorithm runs inside a MATLAB Function block, so all arrays must be compile-time fixed-size — no `find()`, `sort()`, `diff()`, or variable-length logical indexing.
- **Direction independence**: the algorithm must treat all headings equally with no built-in "north = forward" assumption.
- **Oscillation prevention**: without memory between frames, the output can jitter on straight lines or flip at corners.

---

## II. Problem Statement

Given per-frame R, G, B arrays of size 120×160 (uint8), detect a red line in the image and output a unit vector (X, Y) indicating the direction the drone should steer to follow the line. The algorithm must:

1. Detect red pixels reliably using thresholds: R > 180, G < 100, B < 100.
2. Determine the line's direction relative to the drone.
3. Handle turns from gentle curves to extreme 150° corners.
4. Operate with only fixed-size arrays (Simulink constraint).
5. Be direction-independent — no axis is privileged as "forward."

[TODO: Add any images of the arena, line layout, or drone setup if available.]

---

## III. Related Work

- **Centroid-based line following**: Common in ground robots — compute the centroid of detected line pixels and steer toward it. Simple but fails at sharp turns where the centroid averages both branches.
- **Hough Line Transform**: Detects line parameters (ρ, θ) but requires variable-size arrays and is computationally heavy for real-time drone control.
- **Strip-based methods**: Divide the image into horizontal strips and compute per-strip centroids. The tangent between adjacent centroids gives the line direction. Used in many line-following competitions.
- **PID controllers**: Often used downstream of the vision algorithm to smooth steering. Not used here since the output is a direction vector, not a motor command.

[TODO: Add any specific papers or references you consulted.]

---

## IV. Initial Attempts

### Attempt 1: 7×7 Grid Scanner (`lineFollower.m`)

The image was cropped to 119×119 and divided into a 7×7 grid of 17×17 blocks. Each block was classified as "line present" if the red pixel count exceeded a threshold. Starting from the centre block (4,4), a compass-based scan searched for the next line block in 8 directions, prioritizing the last known heading.

**Strengths:** Simple, intuitive, works for moderate turns.

**Weaknesses:**
- Quantized to 8 directions (45° resolution) — poor angle accuracy.
- Forward-cone bug: the scan included ±135° directions, causing U-turns.
- Threshold `0.90 × 17 × 17` was too high; blocks rarely triggered.

These were fixed iteratively (forward cone limited to ±90°, threshold reduced to 10%), but fundamental 45° quantization limited accuracy.

### Attempt 2: 3-Strip Tangent Method (`lineFollower2.m`)

The image was divided into 3 horizontal strips of 40 rows each. The column centroid of red pixels in each strip was computed, and the tangent vector between the top and bottom centroids gave the line direction.

**Strengths:** Much better angle accuracy than the grid. Simple and fast.

**Weaknesses:** Fails on steep turns (>60°) where the line runs vertically — horizontal strips see no lateral offset.

### Attempt 3: Dual-Axis Strips + Heading Projection (`lineFollower3.m`, early version)

Added 8 vertical strips alongside 6 horizontal strips (14 centroids total). Used heading-based projection to identify "farthest forward" and "farthest behind" points.

**Problem found:** At a 150° corner, ALL centroids project negatively onto the heading. The "max projection" picks the old line's centroid → drone steers backward.

---

## V. Final Approach

### Algorithm: Global + Perpendicular Centroid with Heading-Aware Clamping (`lineFollower3.m`)

The final algorithm uses three weighted centroid computations and blends them based on how much of the line is ahead vs. behind.

#### Step 1 — Red Mask
```matlab
mask = double((R > 180) & (G < 100) & (B < 100));
```
A binary 120×160 mask of red pixels. All subsequent computations use this mask with fixed-size matrix operations.

#### Step 2 — Global Centroid
Compute the centroid of ALL red pixels using column/row projection sums:
```
gcCol = Σ(colProj × colIdx) / totalRed
gcRow = Σ(rowProj × rowIdx) / totalRed
```
This is the "centre of mass" of the visible line.

#### Step 3 — Ahead Centroid
For each pixel, compute its signed projection onto the persistent heading vector (dirX, dirY):
```
proj(i,j) = (imgCR - i)·dirX + (j - imgCC)·dirY
```
Pixels with proj > 0 are "ahead" of the drone. The weighted centroid of `mask × max(0, proj + 1)` gives the ahead centroid. On straight lines, this dominates steering.

#### Step 4 — Perpendicular Centroid
Compute each pixel's perpendicular distance from the heading axis:
```
perpDist(i,j) = (imgCR - i)·(-dirY) + (j - imgCC)·dirX
```
Weight by `|perpDist|` to emphasize pixels far from the heading line. At a 150° corner, the new branch sticks out sideways → high perp weight → this centroid points toward the new branch.

#### Step 5 — Adaptive Blending
```
behindRatio = 1 - aheadTot / totalRed
target = (1 - behindRatio)·ahead + behindRatio·perp
```
On straight lines, `behindRatio ≈ 0` → ahead centroid dominates.
At sharp corners, `behindRatio ≈ 0.5–0.8` → perp centroid (new branch) takes over.

#### Step 6 — Angular Clamping (History Buffer)
A circular buffer of size K stores the last K output directions. The raw steering angle is compared against every entry; if the angular difference to **any** historical direction exceeds ±90°, the output is clamped to 90° from that reference. This prevents U-turns regardless of what the centroid computation produces.

#### Step 7 — Smooth Heading Update
```
dir = 0.7·output + 0.3·oldDir   (normalized)
```
The persistent heading blends 70% new output with 30% old heading, providing smooth transitions.

### Key Equations

Steering vector from image centre to blended target:

$$\vec{s} = \begin{pmatrix} \text{imgCR} - \text{blendRow} \\ \text{blendCol} - \text{imgCC} \end{pmatrix}$$

Angular clamping:

$$\Delta\theta = \text{rawAng} - \text{histAng}_k, \quad |\Delta\theta| \leq \text{MAX\_TURN}$$

Heading update:

$$\vec{d}_{new} = \frac{0.7 \cdot \vec{out} + 0.3 \cdot \vec{d}_{old}}{|0.7 \cdot \vec{out} + 0.3 \cdot \vec{d}_{old}|}$$

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Red threshold | R>180, G<100, B<100 | Red pixel detection |
| K (buffer size) | 10 | Number of historical directions for clamping |
| MAX_TURN | π/2 (90°) | Maximum angular deviation from any historical direction |
| α (heading blend) | 0.7 | Weight of new output in heading update |
| Min red pixels | 15 | Below this, keep last heading |

---

## VI. Results and Observation

[TODO: Add test results — describe how the drone performed on straight lines, gentle curves, 90° turns, and 150° corners. Include any screenshots or trajectory plots if available.]

[TODO: Compare the three approaches (grid, 3-strip, final) in terms of angle accuracy and turn handling. A table like:]

| Method | Straight Line | 90° Turn | 150° Turn | Angle Accuracy |
|--------|--------------|----------|-----------|---------------|
| 7×7 Grid | ✓ | ✓ | ✗ | ~45° |
| 3-Strip | ✓ | ✓ | ✗ | ~5° |
| Final (Global+Perp) | ✓ | ✓ | [TODO] | ~5° |

---

## VII. Future Work

- **Adaptive thresholding**: The fixed R>180/G<100/B<100 threshold may fail under different lighting conditions. An adaptive threshold based on histogram analysis could improve robustness.
- **Predictive steering**: Using velocity estimates to predict where the line will be 2–3 frames ahead, enabling smoother anticipation of turns.
- **T-junction handling**: The current algorithm does not handle T-intersections. Adding fork detection and branch commitment logic would be needed for more complex track layouts.
- **PID-based smoothing**: Integrating a PID controller on top of the vision output could reduce residual jitter.

---

## Conclusion

A direction-independent, fixed-size-array-compatible line-following algorithm was developed for the Parrot Mambo minidrone. The final approach uses global and perpendicular centroid blending with a history-based angular clamp to handle turns up to 150°. Three progressively refined approaches were explored, with each iteration addressing specific failure modes discovered during testing. The algorithm is fully compatible with Simulink MATLAB Function blocks and requires no variable-size operations.

[TODO: Add any final observations about the overall experience, difficulties faced, and lessons learned.]

---

## References

[1] [TODO: Add references to any line-following papers or resources you consulted.]

[2] MathWorks, "Parrot Minidrone Support from Simulink," MATLAB Documentation.

[3] [TODO: Add any other references.]

---

*Acknowledgements: [TODO: Add anyone who helped — seniors, teammates, etc.]*
