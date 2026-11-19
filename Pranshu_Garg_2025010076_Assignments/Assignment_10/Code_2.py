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
# Otsu's Global Thresholding Algorithm
# ---------------------------------------------------------
# 1. Compute normalized histogram probabilities p_i for i in [0, 255]
hist = np.bincount(gray_img.ravel(), minlength=256)
prob = hist / gray_img.size
intensities = np.arange(256, dtype=np.float64)

# 2. Cumulative class probabilities omega_1(t) and cumulative means m(t)
omega_1 = np.cumsum(prob)
omega_2 = 1.0 - omega_1
cum_mean = np.cumsum(intensities * prob)
global_mean = cum_mean[-1]

# 3. Compute between-class variance sigma_B^2(t) for all candidate thresholds t
valid = (omega_1 > 0) & (omega_2 > 0)
sigma_b_sq = np.zeros(256, dtype=np.float64)
sigma_b_sq[valid] = ((global_mean * omega_1[valid] - cum_mean[valid]) ** 2) / (
    omega_1[valid] * omega_2[valid]
)

# 4. Optimal threshold maximizes between-class variance
max_variance = np.max(sigma_b_sq)
optimal_threshold = int(np.round(np.mean(np.where(sigma_b_sq == max_variance)[0])))
print(f"Otsu's Optimal Global Threshold (T*): {optimal_threshold}")

# 5. Segment the image into foreground (255) and background (0)
segmented_img = np.where(gray_img > optimal_threshold, 255, 0).astype(np.uint8)

# Save output
cv2.imwrite('output_otsu_segmented.jpg', segmented_img)

# Display original image, histogram with optimal threshold, and segmented image
plt.figure(figsize=(13, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.plot(hist, color='black')
plt.axvline(optimal_threshold, color='crimson', linestyle='--', linewidth=1.8,
            label=f"Otsu T* = {optimal_threshold}")
plt.title("Histogram & Otsu's Threshold")
plt.xlabel('Intensity Level (0–255)')
plt.ylabel('Pixel Count')
plt.xlim([0, 255])
plt.legend(loc='upper right')

plt.subplot(1, 3, 3)
plt.imshow(segmented_img, cmap='gray', vmin=0, vmax=255)
plt.title(f"Otsu Segmented Image (T* = {optimal_threshold})")
plt.axis('off')

plt.tight_layout()
plt.show()