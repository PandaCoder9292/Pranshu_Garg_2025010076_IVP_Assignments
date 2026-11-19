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
# Histogram Equalization Steps
# ---------------------------------------------------------
# 1. Compute histogram (frequency of each intensity level 0..255)
hist = np.bincount(gray_img.ravel(), minlength=256)

# 2. Compute Probability Density Function (PDF)
pdf = hist / gray_img.size

# 3. Compute Cumulative Distribution Function (CDF)
cdf = np.cumsum(pdf)

# 4. Map CDF to new intensity levels [0, 255]
equalized_lut = np.round(255 * cdf).astype(np.uint8)

# 5. Apply lookup table to transform the input image
equalized_img = equalized_lut[gray_img]

# Save outputs
cv2.imwrite('original_grayscale.jpg', gray_img)
cv2.imwrite('output_histogram_equalized.jpg', equalized_img)

# Display original and equalized images alongside their histograms
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(equalized_img, cmap='gray', vmin=0, vmax=255)
plt.title('Histogram Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.plot(hist, color='black')
plt.title('Original Histogram')
plt.xlabel('Intensity Level')
plt.ylabel('Pixel Count')
plt.xlim([0, 255])

plt.subplot(2, 2, 4)
eq_hist = np.bincount(equalized_img.ravel(), minlength=256)
plt.plot(eq_hist, color='black')
plt.title('Equalized Histogram')
plt.xlabel('Intensity Level')
plt.ylabel('Pixel Count')
plt.xlim([0, 255])

plt.tight_layout()
plt.show()