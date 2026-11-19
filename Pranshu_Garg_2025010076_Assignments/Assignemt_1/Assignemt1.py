import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Fetch a random image from the internet (Picsum serves a different image each run)
url = "https://picsum.photos/500/500"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as response:
    image_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)

# Decode the downloaded bytes directly into a grayscale image
gray_img = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)

if gray_img is None:
    raise RuntimeError("Failed to download or decode the random image.")

# ---------------------------------------------------------
# Case 1: Mean intensity of the image as the threshold value
# ---------------------------------------------------------
mean_thresh = np.mean(gray_img)
print(f"Case 1 - Calculated Mean Intensity Threshold: {mean_thresh:.2f}")

binary_case1 = np.where(gray_img >= mean_thresh, 255, 0).astype(np.uint8)

# ---------------------------------------------------------
# Case 2: Threshold value taken from user input
# ---------------------------------------------------------
user_thresh = int(input("Case 2 - Enter threshold value (0 to 255): "))

if not (0 <= user_thresh <= 255):
    raise ValueError("Threshold value must be between 0 and 255.")

binary_case2 = np.where(gray_img >= user_thresh, 255, 0).astype(np.uint8)

# Save the original and output images locally
cv2.imwrite('random_grayscale.jpg', gray_img)
cv2.imwrite('output_case1_mean.jpg', binary_case1)
cv2.imwrite('output_case2_user.jpg', binary_case2)

# Display the original and resulting binary images
plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale (Random)')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(binary_case1, cmap='gray', vmin=0, vmax=255)
plt.title(f'Case 1: Mean Threshold ({mean_thresh:.1f})')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(binary_case2, cmap='gray', vmin=0, vmax=255)
plt.title(f'Case 2: User Threshold ({user_thresh})')
plt.axis('off')

plt.tight_layout()
plt.show()