import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

def fetch_random_grayscale(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response:
        image_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Failed to download or decode image from {url}")
    return img

# Fetch two distinct random grayscale images (Input and Reference)
src_img = fetch_random_grayscale("https://picsum.photos/500/500?random=1")
ref_img = fetch_random_grayscale("https://picsum.photos/500/500?random=2")

# ---------------------------------------------------------
# Histogram Matching via Histogram Equalization
# ---------------------------------------------------------
# 1. Equalize the Input (Source) image histogram -> s_k = T(r_k)
src_hist = np.bincount(src_img.ravel(), minlength=256)
src_cdf = np.cumsum(src_hist) / src_img.size
src_eq_levels = np.round(255 * src_cdf)

# 2. Equalize the Reference image histogram -> v_q = G(z_q)
ref_hist = np.bincount(ref_img.ravel(), minlength=256)
ref_cdf = np.cumsum(ref_hist) / ref_img.size
ref_eq_levels = np.round(255 * ref_cdf)

# 3. Build Lookup Table: for each input intensity k, find z_q minimizing |G(z_q) - T(r_k)|
matching_lut = np.zeros(256, dtype=np.uint8)
for k in range(256):
    closest_z = np.argmin(np.abs(ref_eq_levels - src_eq_levels[k]))
    matching_lut[k] = closest_z

# 4. Map input image pixels to matched intensities
matched_img = matching_lut[src_img]

# Save outputs
cv2.imwrite('input_source.jpg', src_img)
cv2.imwrite('reference_target.jpg', ref_img)
cv2.imwrite('output_histogram_matched.jpg', matched_img)

# Display Input, Reference, and Matched images along with their histograms
plt.figure(figsize=(13, 8))

plt.subplot(2, 3, 1)
plt.imshow(src_img, cmap='gray', vmin=0, vmax=255)
plt.title('Input (Source) Image')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(ref_img, cmap='gray', vmin=0, vmax=255)
plt.title('Reference Image')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(matched_img, cmap='gray', vmin=0, vmax=255)
plt.title('Histogram Matched Image')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.plot(src_hist, color='black')
plt.title('Input Histogram')
plt.xlabel('Intensity')
plt.ylabel('Pixel Count')
plt.xlim([0, 255])

plt.subplot(2, 3, 5)
plt.plot(ref_hist, color='black')
plt.title('Reference Histogram')
plt.xlabel('Intensity')
plt.xlim([0, 255])

plt.subplot(2, 3, 6)
matched_hist = np.bincount(matched_img.ravel(), minlength=256)
plt.plot(matched_hist, color='black')
plt.title('Matched Histogram')
plt.xlabel('Intensity')
plt.xlim([0, 255])

plt.tight_layout()
plt.show()