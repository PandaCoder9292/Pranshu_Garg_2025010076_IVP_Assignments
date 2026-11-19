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

# Convert to float32 to preserve negative values during derivative calculations
img_float = gray_img.astype(np.float32)

# ---------------------------------------------------------
# 1. Sharpening using Laplacian Filter
# ---------------------------------------------------------
laplacian_kernel = np.array([[0,  1, 0],
                             [1, -4, 1],
                             [0,  1, 0]], dtype=np.float32)

laplacian_edges = cv2.filter2D(img_float, -1, laplacian_kernel)
# Subtract Laplacian response because the center coefficient is negative
sharpened_laplacian = np.clip(img_float - laplacian_edges, 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# 2. Sharpening using Sobel Gradient and Canny Edge Detector
# ---------------------------------------------------------
# 2a. Sobel Gradient Operator
sobel_x_kernel = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=np.float32)

sobel_y_kernel = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=np.float32)

grad_x = cv2.filter2D(img_float, -1, sobel_x_kernel)
grad_y = cv2.filter2D(img_float, -1, sobel_y_kernel)

sobel_mag = np.sqrt(grad_x**2 + grad_y**2)
# Normalize gradient magnitude to [0, 255] and add weighted edges to original
sobel_mag_norm = (sobel_mag / (np.max(sobel_mag) + 1e-8)) * 255.0
sharpened_sobel = np.clip(img_float + 0.5 * sobel_mag_norm, 0, 255).astype(np.uint8)

# 2b. Canny Edge Detector
canny_edges = cv2.Canny(gray_img, threshold1=100, threshold2=200)
# Add weighted Canny edges back to the original image to enhance edge sharpness
sharpened_canny = np.clip(img_float + 0.5 * canny_edges.astype(np.float32), 0, 255).astype(np.uint8)

# ---------------------------------------------------------
# 3. Unsharp Masking (Gaussian Filter: size = 5x5, sigma = 3)
# ---------------------------------------------------------
k_size = 5
sigma = 3.0
r = k_size // 2
y, x = np.meshgrid(np.arange(-r, r + 1), np.arange(-r, r + 1), indexing='ij')

gaussian_kernel = np.exp(-(x**2 + y**2) / (2.0 * sigma**2)).astype(np.float32)
gaussian_kernel /= np.sum(gaussian_kernel)

blurred_img = cv2.filter2D(img_float, -1, gaussian_kernel)
unsharp_mask = img_float - blurred_img
sharpened_unsharp = np.clip(img_float + 1.0 * unsharp_mask, 0, 255).astype(np.uint8)

# Save outputs
cv2.imwrite('output_1_laplacian.jpg', sharpened_laplacian)
cv2.imwrite('output_2a_sobel_sharpened.jpg', sharpened_sobel)
cv2.imwrite('output_2b_canny_edges.jpg', canny_edges)
cv2.imwrite('output_2b_canny_sharpened.jpg', sharpened_canny)
cv2.imwrite('output_3_unsharp_masking.jpg', sharpened_unsharp)

# Display original and sharpened results
plt.figure(figsize=(13, 8))

plt.subplot(2, 3, 1)
plt.imshow(gray_img, cmap='gray', vmin=0, vmax=255)
plt.title('Original Grayscale')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(sharpened_laplacian, cmap='gray', vmin=0, vmax=255)
plt.title('1. Laplacian Sharpened')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(sharpened_sobel, cmap='gray', vmin=0, vmax=255)
plt.title('2a. Sobel Gradient Sharpened')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(canny_edges, cmap='gray', vmin=0, vmax=255)
plt.title('2b. Canny Edge Map')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(sharpened_canny, cmap='gray', vmin=0, vmax=255)
plt.title('2b. Canny Sharpened')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(sharpened_unsharp, cmap='gray', vmin=0, vmax=255)
plt.title('3. Unsharp Masking (5x5, σ=3)')
plt.axis('off')

plt.tight_layout()
plt.show()