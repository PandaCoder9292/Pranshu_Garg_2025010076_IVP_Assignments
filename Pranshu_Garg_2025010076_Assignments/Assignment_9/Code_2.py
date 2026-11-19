import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fetch a random image from the internet and decode as grayscale
url = "https://picsum.photos/500/500"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as response:
    image_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)

gray_img = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
if gray_img is None:
    raise RuntimeError("Failed to download or decode the random image.")

# User input for window size (3 or 5)
k = int(input("Enter filter window size (3 or 5): "))
if k not in (3, 5):
    raise ValueError("Window size must be 3 or 5.")

img_float = gray_img.astype(np.float64)

# ---------------------------------------------------------
# 1. Corrupt Image with Gaussian and Speckle Noise
# ---------------------------------------------------------
np.random.seed(42)

# Additive Gaussian Noise: g(x,y) = f(x,y) + N(0, sigma^2)
gauss_noise = np.random.normal(loc=0.0, scale=25.0, size=gray_img.shape)
noisy_gaussian = np.clip(img_float + gauss_noise, 0, 255).astype(np.uint8)

# Multiplicative Speckle Noise: g(x,y) = f(x,y) + f(x,y) * N(0, var)
speckle_noise = np.random.normal(loc=0.0, scale=0.2, size=gray_img.shape)
noisy_speckle = np.clip(img_float + img_float * speckle_noise, 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# 2. Define the Four Denoising Filters (k x k window)
# ---------------------------------------------------------
def wiener_filter(image, win_size):
    img = image.astype(np.float64)
    local_mean = cv2.boxFilter(img, ddepth=-1, ksize=(win_size, win_size), normalize=True)
    local_sq_mean = cv2.boxFilter(img**2, ddepth=-1, ksize=(win_size, win_size), normalize=True)
    local_var = np.maximum(0.0, local_sq_mean - local_mean**2)
    noise_var = np.mean(local_var)

    gain = np.maximum(0.0, local_var - noise_var) / (np.maximum(local_var, noise_var) + 1e-8)
    denoised = local_mean + gain * (img - local_mean)
    return np.clip(denoised, 0, 255).astype(np.uint8)

def apply_all_filters(noisy_img, win_size):
    w_out = wiener_filter(noisy_img, win_size)
    m_out = cv2.medianBlur(noisy_img, win_size)
    g_out = cv2.GaussianBlur(noisy_img, (win_size, win_size), sigmaX=1.5)
    b_out = cv2.bilateralFilter(noisy_img, d=win_size, sigmaColor=50, sigmaSpace=50)
    return w_out, m_out, g_out, b_out

def compute_psnr(original, processed):
    mse = np.mean((original.astype(np.float64) - processed.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)

# Apply filters to both corrupted images
gauss_wiener, gauss_median, gauss_gauss, gauss_bilateral = apply_all_filters(noisy_gaussian, k)
speck_wiener, speck_median, speck_gauss, speck_bilateral = apply_all_filters(noisy_speckle, k)

# Print PSNR comparison table in console
print(f"\n--- PSNR Comparison (dB) with {k}x{k} Window ---")
for label, noisy, w, m, g, b in [
    ("Gaussian Noise", noisy_gaussian, gauss_wiener, gauss_median, gauss_gauss, gauss_bilateral),
    ("Speckle Noise ", noisy_speckle, speck_wiener, speck_median, speck_gauss, speck_bilateral),
]:
    print(f"{label} -> Noisy: {compute_psnr(gray_img, noisy):.2f} | "
          f"Wiener: {compute_psnr(gray_img, w):.2f} | "
          f"Median: {compute_psnr(gray_img, m):.2f} | "
          f"Gaussian: {compute_psnr(gray_img, g):.2f} | "
          f"Bilateral: {compute_psnr(gray_img, b):.2f}")

# Display Comparison Grid
fig, axes = plt.subplots(2, 6, figsize=(18, 7))

row_data = [
    ("Gaussian", noisy_gaussian, gauss_wiener, gauss_median, gauss_gauss, gauss_bilateral),
    ("Speckle",  noisy_speckle,  speck_wiener, speck_median, speck_gauss, speck_bilateral),
]

for r, (noise_name, noisy, w, m, g, b) in enumerate(row_data):
    panels = [
        ("Original", gray_img),
        (f"{noise_name} Noisy\n({compute_psnr(gray_img, noisy):.1f} dB)", noisy),
        (f"(a) Wiener {k}x{k}\n({compute_psnr(gray_img, w):.1f} dB)", w),
        (f"(b) Median {k}x{k}\n({compute_psnr(gray_img, m):.1f} dB)", m),
        (f"(c) Gaussian {k}x{k}\n({compute_psnr(gray_img, g):.1f} dB)", g),
        (f"(d) Bilateral {k}x{k}\n({compute_psnr(gray_img, b):.1f} dB)", b),
    ]
    for c, (title, img_panel) in enumerate(panels):
        axes[r, c].imshow(img_panel, cmap='gray', vmin=0, vmax=255)
        axes[r, c].set_title(title, fontsize=9)
        axes[r, c].axis('off')

plt.tight_layout()
plt.show()