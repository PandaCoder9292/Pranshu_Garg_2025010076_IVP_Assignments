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

# ---------------------------------------------------------
# User Inputs: Initial Window Size and Maximum Window Size (Smax)
# ---------------------------------------------------------
s_init = int(input("Enter initial window size (odd integer >= 3, e.g., 3): "))
s_max = int(input("Enter maximum window size Smax (odd integer >= initial size, e.g., 9): "))

if s_init < 3 or s_init % 2 == 0 or s_max < s_init or s_max % 2 == 0:
    raise ValueError("Both window sizes must be odd integers with 3 <= initial <= Smax.")

# ---------------------------------------------------------
# Corrupt Image with Salt & Pepper (Impulse) Noise (25% density)
# ---------------------------------------------------------
np.random.seed(42)
noisy_img = gray_img.copy()
noise_density = 0.25
rand_matrix = np.random.rand(*gray_img.shape)
noisy_img[rand_matrix < (noise_density / 2.0)] = 0          # Pepper noise
noisy_img[rand_matrix > (1.0 - noise_density / 2.0)] = 255  # Salt noise

# ---------------------------------------------------------
# Adaptive Median Filter Implementation
# ---------------------------------------------------------
def adaptive_median_filter(image, initial_size, max_size):
    z_xy = image.astype(np.int16)
    output = np.zeros_like(z_xy)
    resolved = np.zeros(image.shape, dtype=bool)

    for s in range(initial_size, max_size + 1, 2):
        kernel = np.ones((s, s), dtype=np.uint8)
        z_min = cv2.erode(image, kernel, borderType=cv2.BORDER_REFLECT).astype(np.int16)
        z_max = cv2.dilate(image, kernel, borderType=cv2.BORDER_REFLECT).astype(np.int16)
        z_med = cv2.medianBlur(image, s).astype(np.int16)

        # Stage A: Check if median itself is an impulse (z_min < z_med < z_max)
        stage_a_pass = (z_med - z_min > 0) & (z_med - z_max < 0)
        active_b = (~resolved) & stage_a_pass

        # Stage B: Check if center pixel is an impulse (z_min < z_xy < z_max)
        stage_b_pass = (z_xy - z_min > 0) & (z_xy - z_max < 0)

        # Keep original pixel if not an impulse; otherwise replace with z_med
        output[active_b & stage_b_pass] = z_xy[active_b & stage_b_pass]
        output[active_b & (~stage_b_pass)] = z_med[active_b & (~stage_b_pass)]
        resolved[active_b] = True

        # If Smax is reached, assign z_med to any remaining unresolved pixels
        if s == max_size:
            output[~resolved] = z_med[~resolved]

    return np.clip(output, 0, 255).astype(np.uint8)

# Apply Standard Median Filters and Adaptive Median Filter
median_init = cv2.medianBlur(noisy_img, s_init)
median_max = cv2.medianBlur(noisy_img, s_max)
adaptive_med = adaptive_median_filter(noisy_img, s_init, s_max)

# ---------------------------------------------------------
# Compute PSNR Values
# ---------------------------------------------------------
def compute_psnr(original, restored):
    mse = np.mean((original.astype(np.float64) - restored.astype(np.float64)) ** 2)
    return float('inf') if mse == 0 else 10.0 * np.log10((255.0 ** 2) / mse)

psnr_noisy = compute_psnr(gray_img, noisy_img)
psnr_med_init = compute_psnr(gray_img, median_init)
psnr_med_max = compute_psnr(gray_img, median_max)
psnr_adaptive = compute_psnr(gray_img, adaptive_med)

print("\n--- PSNR Performance Comparison (dB) ---")
print(f"Noisy Image (25% Salt & Pepper)       : {psnr_noisy:.2f} dB")
print(f"Standard Median Filter ({s_init}x{s_init})          : {psnr_med_init:.2f} dB")
print(f"Standard Median Filter ({s_max}x{s_max})          : {psnr_med_max:.2f} dB")
print(f"Adaptive Median Filter ({s_init}x{s_init} -> {s_max}x{s_max}) : {psnr_adaptive:.2f} dB")

# Save outputs
cv2.imwrite('output_noisy_sp.jpg', noisy_img)
cv2.imwrite('output_standard_median.jpg', median_init)
cv2.imwrite('output_adaptive_median.jpg', adaptive_med)

# Display visual comparison
plt.figure(figsize=(15, 4))

panels = [
    ("Original Grayscale", gray_img),
    (f"Noisy Image\nPSNR: {psnr_noisy:.2f} dB", noisy_img),
    (f"Standard Median ({s_init}x{s_init})\nPSNR: {psnr_med_init:.2f} dB", median_init),
    (f"Standard Median ({s_max}x{s_max})\nPSNR: {psnr_med_max:.2f} dB", median_max),
    (f"Adaptive Median ({s_init}→{s_max})\nPSNR: {psnr_adaptive:.2f} dB", adaptive_med),
]

for idx, (title, img_data) in enumerate(panels, start=1):
    plt.subplot(1, 5, idx)
    plt.imshow(img_data, cmap='gray', vmin=0, vmax=255)
    plt.title(title, fontsize=9)
    plt.axis('off')

plt.tight_layout()
plt.show()