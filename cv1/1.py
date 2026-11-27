import numpy as np
import cv2
import matplotlib.pyplot as plt

def kernel(size, sigma):
    r = size // 2
    x = np.arange(-r, r+1)
    y = np.arange(-r, r+1)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

params = [(5, 1.0), (15, 2.0), (25, 3.0)]

for i, (size, sigma) in enumerate(params):

    kernel1 = kernel(size, sigma)

    k = cv2.getGaussianKernel(size, sigma)
    kernel2 = k @ k.T

    plt.subplot(3, 2, 2*i + 1)
    plt.imshow(kernel1, cmap='viridis')
    plt.title(f'manual: {size}x{size}, sigma={sigma}')
    plt.colorbar()

    plt.subplot(3, 2, 2*i + 2)
    plt.imshow(kernel2, cmap='viridis')
    plt.title(f'cv2: {size}x{size}, sigma={sigma}')
    plt.colorbar()

plt.tight_layout()
plt.show()
