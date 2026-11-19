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

# Input gamma value from the user
gamma = float(input("Enter gamma value (e.g., < 1 to brighten, > 1 to darken): "))
if gamma <= 0:
    raise ValueError("Gamma must be greater than 0.")

# Normalize pixel intensities to [0, 1] and apply Power-Law Transform: s = c * (r ^ gamma)
c = 1.0
r_norm = gray_img.astype(np.float32) / 255.0
s_norm = c * np.power(r_norm, gamma)

# Scale back to [0, 255] uint8 format
gamma_img = np.clip(s_norm * 255.0, 0, 255).astype(np.uint8)

# Save output
cv2.imwrite('output_gamma_transform.jpg', gamma_img)

# Display original vs. gamma-transformed image
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(gamma_img, cmap='gray', vmin=0, vmax=255)
plt.title(f'Power-Law Transform (c = 1, γ = {gamma})')
plt.axis('off')

plt.tight_layout()
plt.show()