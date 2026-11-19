import urllib.request
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Optional: scikit-image's random_noise is Python's direct equivalent of MATLAB's imnoise
try:
    from skimage.util import random_noise
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False

# Fetch a random image from the internet and decode as grayscale
url = "https://picsum.photos/500/500"
req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})

with urllib.request.urlopen(req) as response:
    image_bytes = np.asarray(bytearray(response.read()), dtype=np.uint8)

gray_img = cv2.imdecode(image_bytes, cv2.IMREAD_GRAYSCALE)
if gray_img is None:
    raise RuntimeError("Failed to download or decode the random image.")

shape = gray_img.shape
img_float = gray_img.astype(np.float32)

# Helper to draw uniform random variables in (0, 1) avoiding exact 0 for log()
def uniform_01(size):
    return np.random.uniform(1e-10, 1.0, size=size)

# ---------------------------------------------------------
# 1. Uniform Noise: z = a + (b - a) * U
# ---------------------------------------------------------
a_uni, b_uni = -35.0, 35.0
u = uniform_01(shape)
noise_uni_eq = a_uni + (b_uni - a_uni) * u
noise_uni_inbuilt = np.random.uniform(a_uni, b_uni, size=shape)

# ---------------------------------------------------------
# 2. Gaussian Noise (Box-Muller): z = mu + sigma * sqrt(-2*ln(U1)) * cos(2*pi*U2)
# ---------------------------------------------------------
mu, sigma = 0.0, 25.0
u1, u2 = uniform_01(shape), uniform_01(shape)
noise_gauss_eq = mu + sigma * np.sqrt(-2.0 * np.log(u1)) * np.cos(2.0 * np.pi * u2)

if HAS_SKIMAGE:
    # Using skimage.util.random_noise (Python's equivalent of MATLAB's imnoise)
    noisy_sk = random_noise(gray_img, mode='gaussian', mean=mu/255.0, var=(sigma/255.0)**2, clip=False)
    noise_gauss_inbuilt = (noisy_sk * 255.0) - img_float
else:
    noise_gauss_inbuilt = np.random.normal(mu, sigma, size=shape)

# ---------------------------------------------------------
# 3. Erlang (Gamma) Noise: z = -(1/a) * ln(prod_{i=1}^b (1 - U_i))
# ---------------------------------------------------------
a_erl = 0.15   # rate parameter (a > 0)
b_erl = 4      # integer shape parameter (b >= 1)
u_erl = uniform_01((b_erl, *shape))
noise_erl_eq = -(1.0 / a_erl) * np.sum(np.log(1.0 - u_erl), axis=0)
noise_erl_inbuilt = np.random.gamma(shape=b_erl, scale=1.0 / a_erl, size=shape)

# ---------------------------------------------------------
# 4. Exponential Noise: z = -(1/a) * ln(1 - U)
# ---------------------------------------------------------
a_exp = 0.05   # rate parameter (a > 0)
u = uniform_01(shape)
noise_exp_eq = -(1.0 / a_exp) * np.log(1.0 - u)
noise_exp_inbuilt = np.random.exponential(scale=1.0 / a_exp, size=shape)

# ---------------------------------------------------------
# 5. Rayleigh Noise: z = a + sqrt(-b * ln(1 - U))
# ---------------------------------------------------------
a_ray, b_ray = 0.0, 800.0
u = uniform_01(shape)
noise_ray_eq = a_ray + np.sqrt(-b_ray * np.log(1.0 - u))
# NumPy's rayleigh scale parameter relates to Gonzalez & Woods 'b' via scale = sqrt(b / 2)
noise_ray_inbuilt = a_ray + np.random.rayleigh(scale=np.sqrt(b_ray / 2.0), size=shape)

# ---------------------------------------------------------
# Combine Results, Add Noise to Image, and Plot Comparisons
# ---------------------------------------------------------
noise_models = [
    ("1. Uniform Noise", noise_uni_eq, noise_uni_inbuilt),
    ("2. Gaussian Noise", noise_gauss_eq, noise_gauss_inbuilt),
    ("3. Erlang Noise", noise_erl_eq, noise_erl_inbuilt),
    ("4. Exponential Noise", noise_exp_eq, noise_exp_inbuilt),
    ("5. Rayleigh Noise", noise_ray_eq, noise_ray_inbuilt),
]

fig, axes = plt.subplots(5, 3, figsize=(13, 18))

for row, (title, n_eq, n_inbuilt) in enumerate(noise_models):
    noisy_img_eq = np.clip(img_float + n_eq, 0, 255).astype(np.uint8)
    noisy_img_inbuilt = np.clip(img_float + n_inbuilt, 0, 255).astype(np.uint8)

    # Column 1: Noisy image from derived PDF equation
    axes[row, 0].imshow(noisy_img_eq, cmap='gray', vmin=0, vmax=255)
    axes[row, 0].set_title(f"{title}\n(Derived Equation)")
    axes[row, 0].axis('off')

    # Column 2: Noisy image from inbuilt function
    axes[row, 1].imshow(noisy_img_inbuilt, cmap='gray', vmin=0, vmax=255)
    axes[row, 1].set_title(f"{title}\n(Inbuilt Function)")
    axes[row, 1].axis('off')

    # Column 3: Histogram comparison of pure noise distributions
    axes[row, 2].hist(n_eq.ravel(), bins=60, density=True, alpha=0.6, color='steelblue', label='Derived Eq.')
    axes[row, 2].hist(n_inbuilt.ravel(), bins=60, density=True, histtype='step', linewidth=1.8, color='crimson', label='Inbuilt')
    axes[row, 2].set_title(f"{title} Histogram")
    axes[row, 2].set_xlabel("Noise Intensity")
    axes[row, 2].set_ylabel("Probability Density")
    axes[row, 2].legend(loc='upper right')

plt.tight_layout()
plt.show()