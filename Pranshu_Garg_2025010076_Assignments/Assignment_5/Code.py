import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

# Fetch a random image from the internet and decode as grayscale
url = "https://picsum.photos/500/500"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as response:
    image_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)

gray_img = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
if gray_img is None:
    raise RuntimeError("Failed to download or decode the random image.")

# ---------------------------------------------------------
# User Inputs: Filter Size, Sigma, and Padding Method
# ---------------------------------------------------------
k = int(input("Enter filter size (odd integer >= 3, e.g., 3, 5, 7): "))
if k < 3 or k % 2 == 0:
    raise ValueError("Filter size must be an odd integer >= 3.")

sigma = float(input("Enter sigma for Gaussian filter (e.g., 1.0, 2.0): "))
if sigma <= 0:
    raise ValueError("Sigma must be greater than 0.")

print("Available padding methods: zero, replication, reflect")
pad_choice = input("Enter padding method: ").strip().lower()

pad_map = {
    'zero': ('constant', {'constant_values': 0}),
    'replication': ('edge', {}),
    'reflect': ('reflect', {})
}

if pad_choice not in pad_map:
    raise ValueError("Invalid padding method. Choose from: zero, replication, reflect.")

# ---------------------------------------------------------
# Construct the Three Smoothing Kernels (Centered at (0, 0))
# ---------------------------------------------------------
r = k // 2
y, x = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')

# 1. Un-weighted Averaging (Box) Filter
kernel_avg = np.ones((k, k), dtype=np.float32) / (k * k)

# 2. Weighted Filter: h(x, y) = max(|x|, |y|) (normalized to preserve brightness)
raw_weighted = np.maximum(np.abs(x), np.abs(y)).astype(np.float32)
kernel_weighted = raw_weighted / np.sum(raw_weighted)

# 3. Gaussian Filter: h(x, y) = exp(-(x^2 + y^2) / (2 * sigma^2))
raw_gaussian = np.exp(-(x**2 + y**2) / (2.0 * (sigma ** 2))).astype(np.float32)
kernel_gaussian = raw_gaussian / np.sum(raw_gaussian)

# ---------------------------------------------------------
# Apply Padding and 2D Spatial Filtering
# ---------------------------------------------------------
def apply_spatial_filter(image, kernel, pad_mode, pad_kwargs):
    pad_width = kernel.shape[0] // 2
    padded = np.pad(image.astype(np.float32), pad_width=pad_width, mode=pad_mode, **pad_kwargs)
    windows = sliding_window_view(padded, kernel.shape)
    filtered = np.einsum('ijkl,kl->ij', windows, kernel)
    return np.clip(filtered, 0, 255).astype(np.uint8)

mode, kwargs = pad_map[pad_choice]

smooth_avg = apply_spatial_filter(gray_img, kernel_avg, mode, kwargs)
smooth_weighted = apply_spatial_filter(gray_img, kernel_weighted, mode, kwargs)
smooth_gaussian = apply_spatial_filter(gray_img, kernel_gaussian, mode, kwargs)

# Print generated kernels for verification
print("\n1. Averaging Kernel:\n", np.round(kernel_avg, 4))
print("\n2. Weighted Kernel h(x,y) = max(|x|, |y|) [Normalized]:\n", np.round(kernel_weighted, 4))
print(f"\n3. Gaussian Kernel (sigma={sigma}) [Normalized]:\n", np.round(kernel_gaussian, 4))

# Save outputs
cv2.imwrite('output_1_averaging.jpg', smooth_avg)
cv2.imwrite('output_2_weighted_max.jpg', smooth_weighted)
cv2.imwrite('output_3_gaussian.jpg', smooth_gaussian)

# Display original and smoothed images
plt.figure(figsize=(10, 9))

plt.subplot(2, 2, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(smooth_avg, cmap='gray', vmin=0, vmax=255)
plt.title(f'1. Averaging ({k}x{k}, {pad_choice})')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(smooth_weighted, cmap='gray', vmin=0, vmax=255)
plt.title(f'2. Weighted max(|x|,|y|) ({k}x{k}, {pad_choice})')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(smooth_gaussian, cmap='gray', vmin=0, vmax=255)
plt.title(f'3. Gaussian ({k}x{k}, σ={sigma}, {pad_choice})')
plt.axis('off')

plt.tight_layout()
plt.show()