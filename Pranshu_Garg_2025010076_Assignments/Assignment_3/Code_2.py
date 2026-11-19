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

# Create a binary version using mean intensity threshold
mean_thresh = np.mean(gray_img)
binary_img = np.where(gray_img >= mean_thresh, 255, 0).astype(np.uint8)

# ---------------------------------------------------------
# Compute Image Complements: s = (L - 1) - r = 255 - r
# ---------------------------------------------------------
complement_gray = 255 - gray_img
complement_binary = 255 - binary_img

# Save outputs
cv2.imwrite('complement_grayscale.jpg', complement_gray)
cv2.imwrite('complement_binary.jpg', complement_binary)

# Display original vs. complement images
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(complement_gray, cmap='gray', vmin=0, vmax=255)
plt.title('Grayscale Complement')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(binary_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Binary')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(complement_binary, cmap='gray', vmin=0, vmax=255)
plt.title('Binary Complement')
plt.axis('off')

plt.tight_layout()
plt.show()