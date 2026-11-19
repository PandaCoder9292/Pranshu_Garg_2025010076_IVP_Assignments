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
# Compute 8-Neighbor (3x3, R=1) Local Binary Pattern (LBP)
# ---------------------------------------------------------
# Pad image by 1 pixel on all sides to handle border pixels
padded = np.pad(gray_img, pad_width=1, mode='edge')
center = padded[1:-1, 1:-1]

# 8 neighbors ordered clockwise starting from top-left
neighbors = [
    padded[0:-2, 0:-2],  # Top-Left     (bit 0 -> 2^0 = 1)
    padded[0:-2, 1:-1],  # Top          (bit 1 -> 2^1 = 2)
    padded[0:-2, 2:  ],  # Top-Right    (bit 2 -> 2^2 = 4)
    padded[1:-1, 2:  ],  # Right        (bit 3 -> 2^3 = 8)
    padded[2:  , 2:  ],  # Bottom-Right (bit 4 -> 2^4 = 16)
    padded[2:  , 1:-1],  # Bottom       (bit 5 -> 2^5 = 32)
    padded[2:  , 0:-2],  # Bottom-Left  (bit 6 -> 2^6 = 64)
    padded[1:-1, 0:-2],  # Left         (bit 7 -> 2^7 = 128)
]

lbp_img = np.zeros_like(gray_img, dtype=np.uint8)
for bit_idx, neighbor in enumerate(neighbors):
    binary_code = (neighbor >= center).astype(np.uint8)
    lbp_img += binary_code * (1 << bit_idx)

# Compute 256-bin LBP texture feature histogram
lbp_hist = np.bincount(lbp_img.ravel(), minlength=256)

# Save output
cv2.imwrite('output_lbp.jpg', lbp_img)

# Display Original Grayscale, LBP Image, and LBP Histogram
plt.figure(figsize=(13, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(lbp_img, cmap='gray', vmin=0, vmax=255)
plt.title('Local Binary Pattern (3x3 LBP)')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.plot(lbp_hist, color='black')
plt.title('LBP Texture Histogram')
plt.xlabel('LBP Code (0–255)')
plt.ylabel('Pixel Frequency')
plt.xlim([0, 255])

plt.tight_layout()
plt.show()