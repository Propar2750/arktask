import cv2
import numpy as np
import matplotlib.pyplot as plt
import sys

# Force UTF-8 output so special characters (→ ✓ ✗ ≈) print correctly on Windows
sys.stdout.reconfigure(encoding='utf-8')

# ──────────────────────────────────────────────────────────────────────────────
# Load the noisy image
# ──────────────────────────────────────────────────────────────────────────────
image = cv2.imread(r'C:\Users\parvc\Documents\College\ARK\Task 2.2.2\noisy.jpg')
if image is None:
    raise FileNotFoundError("Could not load 'noisy.jpg'.")

h, w, c = image.shape
print(f"Image: {w}x{h}, {c} channels, dtype={image.dtype}")

# Convert to float for analysis
img_f = image.astype(np.float64)

# ──────────────────────────────────────────────────────────────────────────────
# 1. Per-channel statistics (global)
#    If noise = original + N(0, σ), a flat region's std ≈ σ
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("1. GLOBAL PER-CHANNEL STATISTICS")
print("=" * 60)
for i, name in enumerate(["Blue", "Green", "Red"]):
    ch = img_f[:, :, i]
    print(f"  {name:5s}: mean={ch.mean():.2f}  std={ch.std():.2f}  "
          f"min={ch.min():.0f}  max={ch.max():.0f}")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Estimate noise from a flat patch
#    Pick a region that looks uniform — compute local stats there.
#    We'll auto-detect the flattest 32x32 patch by finding the one with
#    the lowest gradient magnitude.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. FLAT PATCH NOISE ESTIMATION")
print("=" * 60)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float64)

# Smooth heavily to approximate the "true" signal, then measure deviation
smoothed = cv2.GaussianBlur(gray, (15, 15), sigmaX=5)

# Find the 32x32 patch with smallest gradient (flattest region)
patch_size = 32
min_var = np.inf
best_y, best_x = 0, 0
for y in range(0, h - patch_size, 8):
    for x in range(0, w - patch_size, 8):
        patch = smoothed[y:y+patch_size, x:x+patch_size]
        var = np.var(patch)
        if var < min_var:
            min_var = var
            best_y, best_x = y, x

print(f"  Flattest 32x32 patch at ({best_x}, {best_y})")

# Analyse noise in the flat patch per channel
for i, name in enumerate(["Blue", "Green", "Red"]):
    patch_ch = img_f[best_y:best_y+patch_size, best_x:best_x+patch_size, i]
    smooth_patch = cv2.GaussianBlur(
        img_f[:, :, i], (15, 15), sigmaX=5
    )[best_y:best_y+patch_size, best_x:best_x+patch_size]
    noise_est = patch_ch - smooth_patch
    print(f"  {name:5s} noise: mean={noise_est.mean():+.2f}  std={noise_est.std():.2f}")

# ──────────────────────────────────────────────────────────────────────────────
# 3. Channel correlation — are R, G, B noise independent?
#    Subtract the local smooth from each channel → residual noise.
#    Compute pairwise Pearson correlation of residuals.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. CHANNEL NOISE CORRELATION")
print("=" * 60)

residuals = []
for i in range(3):
    smooth_ch = cv2.GaussianBlur(img_f[:, :, i], (15, 15), sigmaX=5)
    residuals.append((img_f[:, :, i] - smooth_ch).flatten())

names = ["Blue", "Green", "Red"]
for a in range(3):
    for b in range(a + 1, 3):
        corr = np.corrcoef(residuals[a], residuals[b])[0, 1]
        print(f"  corr({names[a]}, {names[b]}) = {corr:.4f}")

print("\n  → If correlations ≈ 0: noise was added independently per channel")
print("  → If correlations ≈ 1: same noise value added to all channels (luminance noise)")

# ──────────────────────────────────────────────────────────────────────────────
# 4. Noise distribution — histogram of residuals
#    If Gaussian → bell curve.  If salt-and-pepper → spikes at extremes.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. NOISE DISTRIBUTION ANALYSIS")
print("=" * 60)

all_residuals = np.concatenate(residuals)
print(f"  Combined residual: mean={all_residuals.mean():.3f}  std={all_residuals.std():.3f}")
print(f"  Skewness = {float(np.mean((all_residuals - all_residuals.mean())**3) / all_residuals.std()**3):.4f}")
print(f"  Kurtosis = {float(np.mean((all_residuals - all_residuals.mean())**4) / all_residuals.std()**4 - 3):.4f}")
print("  → Skewness ≈ 0 and Kurtosis ≈ 0 confirms Gaussian distribution")

# ──────────────────────────────────────────────────────────────────────────────
# 5. Additive vs Multiplicative — does noise σ depend on brightness?
#    Split pixels into brightness bins and compute noise std in each.
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("5. ADDITIVE vs MULTIPLICATIVE NOISE")
print("=" * 60)

smooth_gray = cv2.GaussianBlur(gray, (15, 15), sigmaX=5)
noise_gray = gray - smooth_gray

bins = [0, 50, 100, 150, 200, 256]
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (smooth_gray >= lo) & (smooth_gray < hi)
    if mask.sum() > 100:
        std_val = noise_gray[mask].std()
        print(f"  Brightness [{lo:3d}-{hi:3d}): {mask.sum():6d} pixels, noise std = {std_val:.2f}")

print("\n  → If std is roughly constant across bins: ADDITIVE noise (img + N(0,σ))")
print("  → If std grows with brightness: MULTIPLICATIVE noise (img × N(1,σ))")

# ──────────────────────────────────────────────────────────────────────────────
# 6. FFT frequency analysis — Gaussian noise is white (flat spectrum)
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. FREQUENCY ANALYSIS (FFT)")
print("=" * 60)

f_transform = np.fft.fft2(gray)
f_shift = np.fft.fftshift(f_transform)
magnitude = 20 * np.log(np.abs(f_shift) + 1)

cy, cx = h // 2, w // 2
# Compare average magnitude in low-freq vs high-freq rings
low_mask = np.zeros((h, w), dtype=bool)
high_mask = np.zeros((h, w), dtype=bool)
Y, X = np.ogrid[:h, :w]
r = np.sqrt((X - cx)**2 + (Y - cy)**2)
low_mask = r < min(h, w) * 0.1
high_mask = (r > min(h, w) * 0.3) & (r < min(h, w) * 0.5)

low_avg = magnitude[low_mask].mean()
high_avg = magnitude[high_mask].mean()
ratio = high_avg / low_avg

print(f"  Low-freq avg magnitude:  {low_avg:.2f}")
print(f"  High-freq avg magnitude: {high_avg:.2f}")
print(f"  Ratio (high/low):        {ratio:.4f}")
print("  → Elevated high-freq ratio confirms white (Gaussian) noise")

# ──────────────────────────────────────────────────────────────────────────────
# 7. SUMMARY — likely noise model
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. SUMMARY — LIKELY NOISE MODEL")
print("=" * 60)

corr_avg = np.mean([np.corrcoef(residuals[a], residuals[b])[0, 1]
                     for a in range(3) for b in range(a + 1, 3)])

stds_by_bin = []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (smooth_gray >= lo) & (smooth_gray < hi)
    if mask.sum() > 100:
        stds_by_bin.append(noise_gray[mask].std())

std_variation = max(stds_by_bin) - min(stds_by_bin) if stds_by_bin else 0

sigma_est = all_residuals.std()

print(f"  Estimated σ (noise strength): {sigma_est:.2f}")
print(f"  Channel correlation (avg):    {corr_avg:.4f}")
print(f"  Additive check (std range):   {std_variation:.2f}")
print()

if abs(corr_avg) < 0.3:
    print("  ✓ Noise is INDEPENDENT per R, G, B channel")
else:
    print("  ✗ Noise is CORRELATED across channels")

if std_variation < sigma_est * 0.5:
    print("  ✓ Noise is ADDITIVE (constant σ across brightness)")
else:
    print("  ✗ Noise appears MULTIPLICATIVE (σ varies with brightness)")

kurt = float(np.mean((all_residuals - all_residuals.mean())**4) / all_residuals.std()**4 - 3)
if abs(kurt) < 1.5:
    print("  ✓ Distribution is GAUSSIAN (kurtosis ≈ 0)")
else:
    print(f"  ? Distribution has excess kurtosis = {kurt:.2f}")

print(f"\n  → Most likely model:  noisy_pixel = original_pixel + N(0, {sigma_est:.1f})")
print(f"     applied INDEPENDENTLY to each of R, G, B")
print(f"\n  → Best removal strategy: denoise each channel separately")
print(f"     (e.g. NL-Means in YCrCb space with h ≈ {sigma_est:.0f})")

# ──────────────────────────────────────────────────────────────────────────────
# Plot residual histogram and save
# ──────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
colors = ['blue', 'green', 'red']
for i, (name, color) in enumerate(zip(names, colors)):
    axes[i].hist(residuals[i], bins=100, density=True, alpha=0.7, color=color, label=name)
    # Overlay fitted Gaussian
    x = np.linspace(residuals[i].min(), residuals[i].max(), 200)
    mu, sig = residuals[i].mean(), residuals[i].std()
    axes[i].plot(x, 1/(sig*np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sig)**2),
                 'k--', linewidth=2, label=f'N({mu:.1f}, {sig:.1f}²)')
    axes[i].set_title(f'{name} channel noise')
    axes[i].legend()
    axes[i].set_xlabel('Residual value')

plt.suptitle('Noise Distribution per Channel (with Gaussian fit)', fontsize=13)
plt.tight_layout()
plt.savefig('noise_analysis.png', dpi=150)
print("\nNoise histograms saved → noise_analysis.png")
plt.show()