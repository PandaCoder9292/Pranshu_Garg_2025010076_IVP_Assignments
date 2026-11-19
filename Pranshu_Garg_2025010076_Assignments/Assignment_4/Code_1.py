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

# Normalize pixel intensities to [0, 1] and apply Log Transform: s = c * log(1 + r)
c = 1.0
r_norm = gray_img.astype(np.float32) / 255.0
s_norm = c * np.log(1.0 + r_norm)

# Scale back to [0, 255] uint8 format
log_img = np.clip(s_norm * 255.0, 0, 255).astype(np.uint8)

# Save output
cv2.imwrite('output_log_transform.jpg', log_img)

# Display original vs. log-transformed image
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(log_img, cmap='gray', vmin=0, vmax=255)
plt.title('Log Transform (c = 1)')
plt.axis('off')

plt.tight_layout()
plt.show()