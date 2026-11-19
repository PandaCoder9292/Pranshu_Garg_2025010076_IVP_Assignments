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
# User Inputs for Border Width and Color
# ---------------------------------------------------------
border_width = int(input("Enter border width (in pixels): "))
if border_width < 0:
    raise ValueError("Border width must be a non-negative integer.")

border_color = int(input("Enter border intensity/color (0=Black, 255=White, 1-254=Gray): "))
if not (0 <= border_color <= 255):
    raise ValueError("Border intensity must be between 0 and 255.")

# For a strict binary image, map the border color to 0 (Black) or 255 (White)
binary_border_color = 255 if border_color >= 128 else 0

# ---------------------------------------------------------
# Apply Border (Padding) using NumPy Matrix Construction
# ---------------------------------------------------------
rows, cols = gray_img.shape
new_rows = rows + 2 * border_width
new_cols = cols + 2 * border_width

# 1. Grayscale image with border
padded_gray = np.full((new_rows, new_cols), border_color, dtype=np.uint8)
padded_gray[border_width:border_width + rows, border_width:border_width + cols] = gray_img

# 2. Binary image with border
padded_binary = np.full((new_rows, new_cols), binary_border_color, dtype=np.uint8)
padded_binary[border_width:border_width + rows, border_width:border_width + cols] = binary_img

# Save outputs
cv2.imwrite('padded_grayscale.jpg', padded_gray)
cv2.imwrite('padded_binary.jpg', padded_binary)

# Display results
plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title(f'Original Grayscale ({rows}x{cols})')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.imshow(padded_gray, cmap='gray', vmin=0, vmax=255)
plt.title(f'Padded Grayscale ({new_rows}x{new_cols}, Color={border_color})')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.imshow(binary_img, cmap='gray', vmin=0, vmax=255)
plt.title(f'Original Binary ({rows}x{cols})')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.imshow(padded_binary, cmap='gray', vmin=0, vmax=255)
plt.title(f'Padded Binary ({new_rows}x{new_cols}, Color={binary_border_color})')
plt.axis('off')

plt.tight_layout()
plt.show()