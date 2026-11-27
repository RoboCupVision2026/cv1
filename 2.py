import cv2
import numpy as np
import matplotlib.pyplot as plt

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

params = [(3, 0.6), (7, 1.4), (15, 3.0)]
fig, axes = plt.subplots(3, 3, figsize = (12, 10))

for i, (size, sigma) in enumerate(params):
    k1 = gauss_kernel(size, sigma)
    k = cv2.getGaussianKernel(size, sigma)
    k2 = k @ k.T
    diff = np.abs(k1 - k2)
    print(f"size = {size}, sigma = {sigma}")
    print("Manual kernel:\n", k1)
    print("OpenCV kernel:\n", k2)
    print("Max error:", diff.max())
    print()

    vmax = max(k1.max(), k2.max())
    a = axes[i, 0].imshow(
        k1, cmap = "hot", vmin = 0, vmax = vmax
    )
    axes[i, 0].set_title(
        f"Manual: {size}x{size}, sigma={sigma}"
    )

    b = axes[i, 1].imshow(
        k2, cmap = "hot", vmin = 0, vmax = vmax
    )
    axes[i, 1].set_title("OpenCV")

    c = axes[i, 2].imshow(diff, cmap = "viridis")
    axes[i, 2].set_title("Absolute error")

    fig.colorbar(a, ax = axes[i, 0])
    fig.colorbar(b, ax = axes[i, 1])
    fig.colorbar(c, ax = axes[i, 2])

plt.tight_layout()
plt.savefig("2.png", dpi = 200)
plt.show()



