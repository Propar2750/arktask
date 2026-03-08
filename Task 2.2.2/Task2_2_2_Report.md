# Task 2.2.2 — Noise Analysis and Image Denoising

**Author:** [Your Name], [Other Contributors]

---

## Abstract

This report presents a comprehensive noise analysis and denoising pipeline for a heavily corrupted colour image. The noise was characterized as independent, additive, approximately Gaussian with σ ≈ 63 per channel, and spectrally white. Five denoising methods were implemented and compared: Gaussian blur, bilateral filtering, non-local means, channel-wise YCrCb denoising, and BM3D. Additionally, a custom edge-aware blur using BFS flood-fill was developed to denoise flat regions while strictly preserving edges. The analysis pipeline provides quantitative noise characterization to guide filter selection.

[TODO: Add any additional context about the source of the noisy image.]

---

## I. Introduction

Task 2.2.2 requires denoising a heavily corrupted colour image (`noisy.jpg`). The task involves two phases:

1. **Noise analysis**: Characterize the type, distribution, and magnitude of the noise (additive vs. multiplicative, Gaussian vs. uniform, per-channel correlation, spectral properties).
2. **Denoising**: Apply multiple denoising algorithms, compare results, and identify the best approach.

A key challenge is the very high noise level (σ ≈ 63 out of 255) — standard light-denoising filters are insufficient, while aggressive filtering destroys important details and edges.

---

## II. Problem Statement

Given a noisy colour image (`noisy.jpg`):
1. **Analyze** the noise to determine its type, distribution, standard deviation, and spectral characteristics.
2. **Denoise** the image using multiple methods.
3. **Compare** the denoising methods and identify the most effective approach.
4. **Implement** a custom edge-aware denoising method that preserves structural edges.

[TODO: Add the noisy image here for reference.]

---

## III. Related Work

- **Gaussian Blur**: Simplest approach — convolves with a Gaussian kernel. Fast but blurs edges.
- **Bilateral Filter** [1]: Edge-preserving filter that weights pixels by both spatial distance and colour similarity.
- **Non-Local Means** [2]: Compares patches across the entire image, averaging similar patches. State-of-the-art for Gaussian noise when BM3D is unavailable.
- **BM3D** [3]: Block-matching and 3D collaborative filtering. Groups similar 2D patches into 3D arrays, applies transform-domain shrinkage. Widely considered the best classical denoiser.
- **Channel-wise denoising**: Converts to a perceptual colour space (YCrCb) to denoise luminance and chrominance independently with different strengths.

[1] C. Tomasi and R. Manduchi, "Bilateral Filtering for Gray and Color Images," ICCV 1998.

[2] A. Buades, B. Coll, and J.-M. Morel, "A Non-Local Algorithm for Image Denoising," CVPR 2005.

[3] K. Dabov et al., "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering," IEEE TIP, 2007.

---

## IV. Initial Attempts

### Noise Analysis (`examine.py`)

Before denoising, a comprehensive noise analysis was performed:

1. **Global per-channel statistics**: Mean and standard deviation of each B, G, R channel across the entire image.

2. **Flat-patch noise estimation**: Automatically detected the flattest 32×32 region (lowest local standard deviation) and measured noise σ in that patch — giving a clean estimate uncontaminated by image structure.

3. **Channel correlation**: Checked whether noise in B, G, R channels is independent or correlated by computing inter-channel correlation coefficients.

4. **Distribution analysis**: Fitted a Gaussian distribution to the noise histogram. A good fit confirms Gaussian noise.

5. **Additive vs. multiplicative test**: Compared noise σ in bright vs. dark regions. If σ is constant → additive noise. If σ scales with intensity → multiplicative (Poisson-like) noise.

6. **Frequency analysis (FFT)**: Computed the power spectrum of the noise. A flat spectrum → white noise (equal energy at all frequencies). A peaked spectrum → structured/periodic noise.

**Key findings:**
- Noise is **additive** (constant σ regardless of region brightness)
- Noise is **independent** across channels (low inter-channel correlation)
- Noise is approximately **Gaussian** (good histogram fit)
- σ ≈ **63** per channel (very heavy)
- Noise is **spectrally white** (flat FFT)

**Conclusion**: The noise model is i.i.d. Gaussian N(0, 63) added independently to each R, G, B channel. This is the classic AWGN model, for which Non-Local Means or BM3D are optimal denoising strategies.

---

## V. Final Approach

### Five Denoising Methods (`solution.py`)

#### Method A — Gaussian Blur
```python
cv2.blur(image, (21, 21))  +  cv2.GaussianBlur(..., (7,7))  ×2
```
Simple low-pass filter. Very fast but blurs edges. Used as a baseline.

#### Method B — Bilateral Filter
```python
cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
```
Edge-preserving filter. Weights pixels by both spatial distance AND colour similarity. Better edge preservation than Gaussian blur but slower.

#### Method C — Non-Local Means
```python
cv2.fastNlMeansDenoisingColored(image, None, h=30, hColor=30, templateWindowSize=5, searchWindowSize=21)
```
Patch-based denoising. Compares 5×5 patches across a 21×21 search window and averages similar patches. Excellent for Gaussian noise. Followed by a light 3×3 Gaussian blur to smooth residual artifacts.

#### Method D — YCrCb Channel-wise Denoising
```python
# Convert BGR → YCrCb
# Y (luminance):   fastNlMeansDenoising(h=6)    — light, preserve detail
# Cr (red chroma):  fastNlMeansDenoising(h=15)   — aggressive
# Cb (blue chroma): fastNlMeansDenoising(h=15)   — aggressive
```
Exploits human perception: eyes are more sensitive to luminance detail than colour detail. Denoise luminance lightly (preserve edges) and chrominance aggressively (remove colour noise).

#### Method E — BM3D
```python
bm3d.bm3d(image_rgb, sigma_psd=63/255)
```
State-of-the-art collaborative filtering. Groups similar patches into 3D arrays, applies transform-domain thresholding (hard thresholding in step 1, Wiener filtering in step 2). Best PSNR for Gaussian noise.

### Custom Edge-Aware Blur (`try.py`)

A fully custom implementation that:

1. **Edge detection** — Multi-scale Sobel gradients with non-maximum suppression and hysteresis thresholding. Produces a clean edge map without using `cv2.Canny`.

2. **BFS flood-fill blur** — For each non-edge pixel, BFS explores a kernel_size × kernel_size window. The flood **cannot cross edge pixels**. Only reachable non-edge pixels on the same side of the edge contribute to the average. This ensures that edges are never blurred across.

3. **Edge pixel inpainting** — Edge pixels are iteratively filled from their non-edge neighbours (outside-in), so the final image has no black edge artifacts.

**Parameters (all tweakable):**
| Parameter | Default | Description |
|-----------|---------|-------------|
| `EDGE_PRE_BLUR` | 11 | Gaussian blur before edge detection |
| `SOBEL_KSIZE` | 7 | Sobel kernel size |
| `LOW_PCT` | 60 | Lower hysteresis percentile |
| `HIGH_PCT` | 85 | Upper hysteresis percentile |
| `EDGE_DILATE` | 1 | Edge dilation in pixels |
| `BLUR_KERNEL` | 11 | BFS blur window size |
| `BLUR_ITERS` | 1 | Number of blur passes |

---

## VI. Results and Observation

[TODO: Add the output images for each method (denoised_gaussian.jpg, denoised_bilateral.jpg, denoised_nlmeans.jpg, denoised_channelwise.jpg, denoised_bm3d.jpg, edge_aware_blur.jpg).]

[TODO: Compare methods. Suggested format:]

| Method | Speed | Edge Preservation | Noise Removal | Overall |
|--------|-------|-------------------|---------------|---------|
| A. Gaussian Blur | Fast | Poor | Good | Baseline |
| B. Bilateral Filter | Medium | Good | Medium | Better |
| C. Non-Local Means | Slow | Very Good | Very Good | Strong |
| D. YCrCb Channel-wise | Slow | Very Good | Very Good | Strong |
| E. BM3D | Slow | Excellent | Excellent | Best |
| Custom Edge-Aware | Very Slow | Excellent | Good | Unique |

[TODO: Add PSNR/SSIM values if a ground-truth clean image is available.]

---

## VII. Future Work

- **Deep learning approaches**: Trained denoisers (DnCNN, FFDNet, Restormer) can outperform classical methods, especially at high noise levels.
- **Noise-level estimation**: Automatic σ estimation from the image would make the pipeline fully automatic.
- **GPU acceleration**: The BFS-based edge-aware blur is extremely slow (pixel-by-pixel BFS). A GPU implementation or approximation using guided filtering could achieve similar results orders of magnitude faster.
- **Blind denoising**: Methods that work without knowing σ (e.g., noise2noise, self2self) would be more practical.

---

## Conclusion

A comprehensive noise analysis pipeline confirmed the noise as additive, independent, Gaussian with σ ≈ 63 and spectrally white. Five denoising methods were compared, with BM3D providing the best results for this noise model. A custom edge-aware blur was also developed using BFS flood-fill, demonstrating that edges can be strictly preserved during denoising at the cost of computation time.

[TODO: Add any personal observations and lessons learned.]

---

## References

[1] C. Tomasi and R. Manduchi, "Bilateral Filtering for Gray and Color Images," Proc. IEEE ICCV, pp. 839–846, 1998.

[2] A. Buades, B. Coll, and J.-M. Morel, "A Non-Local Algorithm for Image Denoising," Proc. IEEE CVPR, Vol. 2, pp. 60–65, 2005.

[3] K. Dabov, A. Foi, V. Katkovnik, and K. Egiazarian, "Image Denoising by Sparse 3-D Transform-Domain Collaborative Filtering," IEEE Trans. Image Processing, Vol. 16, No. 8, pp. 2080–2095, 2007.

[4] [TODO: Add any other references.]

---

*Acknowledgements: [TODO: Add anyone who helped.]*
