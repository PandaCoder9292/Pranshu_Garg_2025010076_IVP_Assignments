import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fetch a random color image from the internet
url = "https://picsum.photos/500/500"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as response:
    image_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)

# Decode image (OpenCV loads as BGR) and convert to RGB
bgr_img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
if bgr_img is None:
    raise RuntimeError("Failed to download or decode the random image.")

rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)

# Extract R, G, B planes as float32 to prevent uint8 overflow during addition
R = rgb_img[:, :, 0].astype(np.float32)
G = rgb_img[:, :, 1].astype(np.float32)
B = rgb_img[:, :, 2].astype(np.float32)

# ---------------------------------------------------------
# Case 1: Mean average of the three planes
# ---------------------------------------------------------
gray_case1 = ((R + G + B) / 3.0).astype(np.uint8)

# ---------------------------------------------------------
# Case 2: Weighted average of R, G, B planes from user input
# ---------------------------------------------------------
print("Case 2 - Enter weights for R, G, B (each between 0 and 1, summing to 1):")
w_r = float(input("Enter weightage for Red (R): "))
w_g = float(input("Enter weightage for Green (G): "))
w_b = float(input("Enter weightage for Blue (B): "))

# Validate individual bounds and total sum (with tolerance for floating-point precision)
if not (0 <= w_r <= 1 and 0 <= w_g <= 1 and 0 <= w_b <= 1):
    raise ValueError("Each weightage must be between 0 and 1.")

if abs((w_r + w_g + w_b) - 1.0) > 1e-6:
    raise ValueError(f"Sum of weightages must equal 1 (got {w_r + w_g + w_b:.4f}).")

gray_case2 = (w_r * R + w_g * G + w_b * B).astype(np.uint8)

# Save original and grayscale outputs locally
cv2.imwrite('random_color.jpg', bgr_img)
cv2.imwrite('output_case1_mean_gray.jpg', gray_case1)
cv2.imwrite('output_case2_weighted_gray.jpg', gray_case2)

# Display the original and resulting grayscale images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(rgb_img)
plt.title('Original RGB Image')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(gray_case1, cmap='gray', vmin=0, vmax=255)
plt.title('Case 1: Mean Average')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(gray_case2, cmap='gray', vmin=0, vmax=255)
plt.title(f'Case 2: Weighted ({w_r}, {w_g}, {w_b})')
plt.axis('off')

plt.tight_layout()
plt.show()