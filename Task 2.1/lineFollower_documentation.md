# lineFollower — Detailed Documentation

## Overview

`lineFollower(R, G, B)` is a MATLAB function designed for a Simulink MATLAB Function block that steers a drone to follow a red line. It takes the drone's downward-facing camera image (split into R, G, B channels) and outputs a unit vector `(X, Y)` indicating the direction the drone should move.

**Coordinate convention:**  
- **X** = vertical up / forward  
- **Y** = horizontal right  

---

## Inputs & Outputs

| Name | Size | Type | Description |
|------|------|------|-------------|
| `R` | 120×160 | `uint8` | Red channel |
| `G` | 120×160 | `uint8` | Green channel |
| `B` | 120×160 | `uint8` | Blue channel |
| `X` | scalar | `double` | X-component of unit steering vector (forward = positive) |
| `Y` | scalar | `double` | Y-component of unit steering vector (right = positive) |

---

## Algorithm — Step by Step

### Step 1: Centre-Crop to 119×119

```matlab
r = double(R(1:119, 21:139));
```

The original 120×160 image is cropped to a **119×119** square by:
- Taking **rows 1 to 119** (dropping the last row)
- Taking **columns 21 to 139** (dropping 20 columns from each side)

**Why 119×119?** We need a square that divides evenly into a 7×7 grid. 119 = 7 × 17, so each grid block is exactly **17×17 pixels**. The square crop also means all directions (up/down/left/right) have equal pixel coverage — no directional bias.

**Why centre the crop?** Column 80 of the original image (horizontal centre) maps to column 60 of the cropped image, keeping the drone's position at the image centre.

```
Original 120×160:
┌──────────────────────────────────────┐
│  20 cols  │     119 cols used      │20│
│  dropped  │    (cols 21–139)       │  │
│           │                        │  │
│           │   119 rows (1–119)     │  │
│           │                        │  │
└──────────────────────────────────────┘
  row 120 dropped
```

---

### Step 2: Red Pixel Mask

```matlab
mask = double((r > 180) & (g < 100) & (b < 100));
```

Creates a **binary 119×119 matrix** of 0s and 1s.  
A pixel is red (1) if:
- R channel > 180 (high red)
- G channel < 100 (low green)
- B channel < 100 (low blue)

If fewer than 20 red pixels exist in the entire image, the function returns `X=1, Y=0` (go straight forward) immediately. This prevents the algorithm from chasing noise.

---

### Step 3: Divide into 7×7 Grid & Count Red Pixels

```matlab
counts = zeros(7, 7);
for i = 1:7
    r1 = (i - 1) * 17 + 1;
    for j = 1:7
        c1 = (j - 1) * 17 + 1;
        counts(i, j) = sum(sum(mask(r1:r1+16, c1:c1+16)));
    end
end
```

The 119×119 mask is divided into **49 blocks**, each **17×17 = 289 pixels**.

```
Grid layout (row i, col j):
         col1  col2  col3  col4  col5  col6  col7
row 1:  [1,1] [1,2] [1,3] [1,4] [1,5] [1,6] [1,7]
row 2:  [2,1] [2,2] [2,3] [2,4] [2,5] [2,6] [2,7]
row 3:  [3,1] [3,2] [3,3] [3,4] [3,5] [3,6] [3,7]
row 4:  [4,1] [4,2] [4,3] [4,4] [4,5] [4,6] [4,7]  ← centre row
row 5:  [5,1] [5,2] [5,3] [5,4] [5,5] [5,6] [5,7]
row 6:  [6,1] [6,2] [6,3] [6,4] [6,5] [6,6] [6,7]
row 7:  [7,1] [7,2] [7,3] [7,4] [7,5] [7,6] [7,7]
                              ↑
                         centre col
```

**Block (4,4)** = image centre = drone position.

`counts(i,j)` = number of red pixels in that 17×17 block (range 0 to 289).

All operations use fixed-size slicing (`r1:r1+16, c1:c1+16` = always 17×17), so the code is Simulink code-generation safe.

---

### Step 4: Directional Priority Scan (Ring 1)

**The 8 ring-1 neighbours** of centre block (4,4) are mapped to compass directions:

```
              col3  col4  col5
        row3: [NW]  [ N]  [NE]
        row4: [ W]  [4,4] [ E]
        row5: [SW]  [ S]  [SE]
```

| Direction | Compass | Grid (row, col) | Index |
|-----------|---------|-----------------|-------|
| N (forward/up) | 1 | (3, 4) | North |
| NE | 2 | (3, 5) | North-East |
| E (right) | 3 | (4, 5) | East |
| SE | 4 | (5, 5) | South-East |
| S (backward) | 5 | (5, 4) | South |
| SW | 6 | (5, 3) | South-West |
| W (left) | 7 | (4, 3) | West |
| NW | 8 | (3, 3) | North-West |

**Note on directions:** Row 1 = top of image = farthest forward. Row 7 = bottom = behind the drone. Column 1 = left. Column 7 = right.

#### Scan Priority Order

The scan order depends on `lastDir` — the direction the drone was travelling in the previous frame. The idea: **check forward first, then slightly off-forward, then sideways, then behind.**

```matlab
baseOff   = [0, 1, 7, 2, 6, 3, 5, 4];
scanOrder = mod(lastDir - 1 + baseOff, 8) + 1;
```

`baseOff` defines relative priority:
- `0` = same direction as lastDir (forward)
- `1` = one step clockwise (slight right)
- `7` = one step counter-clockwise (slight left)  (7 mod 8 = -1)
- `2` = two steps clockwise (hard right)
- `6` = two steps counter-clockwise (hard left)
- `3` = three steps clockwise
- `5` = three steps counter-clockwise
- `4` = directly behind (lowest priority)

**Example:** If `lastDir = 1` (North), scan order = N, NE, NW, E, W, SE, SW, S.  
If `lastDir = 3` (East), scan order = E, SE, NE, S, N, SW, NW, W.

The function scans ring-1 blocks in this order. The **first block with `counts(i,j) >= THRESH`** wins.

#### Threshold

```matlab
THRESH = 0.10 * 17 * 17;   % ≈ 29 pixels
```

A block is "active" (has significant red line presence) if at least **10%** of its pixels (≈29 out of 289) are red. This is far more robust than requiring the block to be fully saturated — a thin line passing through a 17×17 block easily hits 29 pixels.

---

### Step 5: Fallback — Max-Count Block

If **no ring-1 block** exceeds the threshold (rare — happens at very sharp turns or when the line is far away), the algorithm falls back to scanning **all 48 non-centre blocks** and picks the one with the highest red pixel count:

```matlab
for i = 1:7
    for j = 1:7
        if (i == 4) && (j == 4), continue; end
        if counts(i, j) > maxCnt
            maxCnt = counts(i, j);
            chosenR = i;  chosenC = j;
        end
    end
end
```

This handles ring-2 and ring-3 blocks without needing 8 separate precomputed scan orders for them. Since going beyond ring-1 means the line is far from the drone, any direction toward maximum red presence is a reasonable recovery strategy.

If even this finds nothing (total silence), return `X=1, Y=0` (go forward).

---

### Step 6: Compute Steering Vector

```matlab
bRow = (chosenR - 1) * 17 + 9;    % block-centre row in cropped image
bCol = (chosenC - 1) * 17 + 9;    % block-centre col in cropped image

steerX = 60.0 - bRow;             % positive X = up = forward
steerY = bCol - 60.0;             % positive Y = right
```

The chosen block's **geometric centre** (not per-pixel centroid) is computed:
- Row centre = `(i-1)*17 + 9` — this is the 9th row of the 17-row block (middle pixel)
- Column centre = `(j-1)*17 + 9` — same logic

The steering vector points from the image centre `(60, 60)` toward this block centre.

**Why block centre instead of per-pixel centroid?**
- Faster (no per-pixel weighted sums)
- With 49 blocks, the 17-pixel resolution is precise enough
- Block centres are compile-time constants, avoiding any variable-size operations

The vector is then normalised to a unit vector:

```matlab
mag = sqrt(steerX^2 + steerY^2);
X = steerX / mag;
Y = steerY / mag;
```

---

### Step 7: Update Direction Memory

```matlab
dr = 4 - chosenR;      % positive = north (up in grid)
dc = chosenC - 4;      % positive = east  (right in grid)
lastDir = mod(round(atan2(dc, dr) / (pi/4)), 8) + 1;
```

`lastDir` (persistent across frames) is updated based on where the chosen block is relative to centre (4,4):

| `dr, dc` | `atan2` angle | `lastDir` |
|-----------|---------------|-----------|
| (1, 0) | 0° | 1 (N) |
| (1, 1) | 45° | 2 (NE) |
| (0, 1) | 90° | 3 (E) |
| (-1, 1) | 135° | 4 (SE) |
| (-1, 0) | 180° | 5 (S) |
| (-1,-1) | -135° | 6 (SW) |
| (0, -1) | -90° | 7 (W) |
| (1, -1) | -45° | 8 (NW) |

For ring-2/3 blocks (e.g., block (1,6)), `atan2` naturally quantises to the nearest compass direction, so the fallback path also updates direction correctly.

---

## Visual Summary

```
Frame input (120×160)
        │
        ▼
  ┌─────────────┐
  │ Crop 119×119 │  (rows 1-119, cols 21-139)
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ Red mask     │  R>180 & G<100 & B<100 → binary 119×119
  └──────┬──────┘
         ▼
  ┌─────────────┐
  │ 7×7 grid    │  Count red pixels per 17×17 block → counts(7,7)
  └──────┬──────┘
         ▼
  ┌─────────────────────────────────┐
  │ Ring-1 scan (direction-aware)   │  Check 8 neighbours of (4,4)
  │ Priority: forward → sides →    │  in order based on lastDir
  │           behind                │  First block ≥ threshold wins
  └──────┬───────────┬──────────────┘
    found│           │not found
         ▼           ▼
  ┌──────────┐  ┌──────────────┐
  │ Use      │  │ Fallback:    │
  │ ring-1   │  │ max-count    │
  │ block    │  │ among all 48 │
  └──────┬───┘  └──────┬───────┘
         └──────┬──────┘
                ▼
  ┌─────────────────────┐
  │ Block centre → unit │  steerX = 60 - bRow
  │ vector from (60,60) │  steerY = bCol - 60
  └──────┬──────────────┘
         ▼
  ┌─────────────────────┐
  │ Update lastDir      │  atan2-based compass quantisation
  └──────┬──────────────┘
         ▼
      (X, Y) output
```

---

## Simulink Compatibility Notes

- **No variable-size arrays**: All slicing uses literal bounds (`r1:r1+16`), all outputs are scalar.
- **No `find`, `sort`, `diff`**: These produce variable-length outputs that break code generation.
- **`persistent lastDir`**: Maintains state across simulation steps. Initialised to 1 (North) on first call.
- **No external toolbox dependencies**: Only basic arithmetic & logical ops.

---

## Tunable Parameters

| Parameter | Current Value | Effect |
|-----------|---------------|--------|
| Red threshold `r >` | 180 | Lower → more sensitive to faded red; higher → stricter |
| Green/Blue ceiling | 100 | Raise if the red is not pure (e.g., orange tones) |
| `THRESH` | 10% of block area (≈29) | Lower → more sensitive, may chase noise; higher → needs strong line presence |
| Minimum pixel count | 20 | Below this the function assumes no line is visible |
| Crop region | rows 1:119, cols 21:139 | Adjust if the camera is offset from drone centre |
