import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def gauss_kernel(size, sigma):
    r = size // 2
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            x = i - r
            y = j - r
            kernel[i, j] = (
                np.exp(-(x * x + y * y) / (2 * sigma * sigma)) / (2 * np.pi * sigma * sigma)
            )
    return kernel / np.sum(kernel)

def replicate_padding(img, r):
    h, w = img.shape[:2]
    result = np.zeros(
        (h + 2 * r, w + 2 * r) + img.shape[2:],
        dtype = img.dtype
    )
    for i in range(h + 2 * r):
        for j in range(w + 2 * r):
            x = min(max(i - r, 0), h - 1)
            y = min(max(j - r, 0), w - 1)
            result[i, j] = img[x, y]
    return result

def filter2d(img, kernel):
    h, w = img.shape[:2]
    size = kernel.shape[0]
    r = size // 2
    padded = replicate_padding(img, r)
    result = np.zeros(img.shape, dtype = np.float64)

    if img.ndim == 3:
        kernel = kernel[:, :, np.newaxis]

    for i in range(h):
        for j in range(w):
            region = padded[i : i + size, j : j + size]
            result[i, j] = np.sum(region * kernel, axis = (0, 1))

    return np.clip(result, 0, 255).astype(np.uint8)

params = [(3, 0.6), (7, 1.4), (15, 3.0)]
img = data.cat().astype(np.uint8)
fig, axes = plt.subplots(2, 2, figsize = (10, 8))
axes = axes.flatten()

axes[0].imshow(img, vmin = 0, vmax = 255)
axes[0].set_title("Original")

for i, (size, sigma) in enumerate(params):
    kernel = gauss_kernel(size, sigma)
    result = filter2d(img, kernel)
    axes[i + 1].imshow(
        result, vmin = 0, vmax = 255
    )
    axes[i + 1].set_title(
        f"Size={size}*{size}, sigma={sigma}"
    )

for ax in axes:
    ax.axis("off")

plt.tight_layout()
plt.savefig("3.png", dpi = 200)
plt.show()