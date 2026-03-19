# ARK Perception Tasks

Submission for ARK (Aerial Robotics Kharagpur) perception tasks — three computer vision and drone control assignments.

**Student ID:** 25HS10043

## Tasks

### Task 26_1 — Drone Line Following (MATLAB)
Line-following algorithm for the Parrot Mambo minidrone using its downward-facing camera. Implemented with fixed-size arrays for Simulink code generation compatibility.

### Task 26_2 — Image Cleaning & Denoising (Python)
- **Task 2.2.1:** Otsu binarization with connected-component cleanup for image cleaning.
- **Task 2.2.2:** Noise analysis and denoising — compares Gaussian, bilateral, non-local means, BM3D, and a custom edge-aware blur filter.

### Task 26_3 — Medial Axis Detection (Python/Jupyter)
Detects the medial axis of surgical tools in video using background subtraction, morphological cleanup, and a **custom Hough Line Transform implemented from scratch**. Outputs annotated video with medial axis overlay.

## Project Structure

```
25HS10043_Perception_Task 26_1/   # MATLAB line-follower scripts
25HS10043_Perception_Task 26_2/   # Python image processing scripts
  Task_2.2.1/                     # Otsu binarization + cleanup
  Task_2.2.2/                     # Noise analysis + denoising
25HS10043_Perception_Task 26_3/   # Jupyter notebook for medial axis pipeline
ARK Perception Task.pdf           # Task specification
```

Each task folder includes a LaTeX report (`.tex`) and compiled PDF.

## Getting Started

```bash
# Set up Python environment
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
pip install opencv-python numpy scipy matplotlib bm3d jupyter

# Run Python scripts
python "25HS10043_Perception_Task 26_2/Task_2.2.1/final_otsu_cleanup.py"
python "25HS10043_Perception_Task 26_2/Task_2.2.2/noise_analysis.py"

# Run Jupyter notebook
jupyter notebook "25HS10043_Perception_Task 26_3/medial_axis_pipeline.ipynb"

# MATLAB files run in MATLAB/Simulink
```

## Dependencies

- **Python:** opencv-python, numpy, scipy, matplotlib, bm3d, jupyter
- **MATLAB/Simulink** (for Task 26_1)
