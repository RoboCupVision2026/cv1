import numpy as np
import cv2
import matplotlib.pyplot as plt


def gaussian_kernel(size, sigma):
    
    if size <= 0 or size % 2 == 0:
        raise ValueError("size 必须是正奇数")
    if sigma <= 0:
        raise ValueError("sigma 必须大于 0")

    centre = size // 2
    kernel = np.zeros((size, size), dtype=float)

    for row in range(size):
        for col in range(size):
            x = col - centre
            y = row - centre

            kernel[row, col] = (
                np.exp(-(x**2 + y**2) / (2 * sigma**2))
                / (2 * np.pi * sigma**2)
            )
    kernel = kernel / kernel.sum()
    return kernel


if __name__ == "__main__":

    cases = [
        (3, 1.0),
        (5, 1.5),
        (9, 2.0),
    ]

    plt.figure(figsize=(10, 12))

    for index in range(len(cases)):
        size, sigma = cases[index]
        
        v1 = gaussian_kernel(size, sigma)

        kernel2 = cv2.getGaussianKernel(size, sigma, cv2.CV_64F)
        vr = kernel2 @ kernel2.T

        max_error = np.abs(v1 - vr).max()

        print(
            f"size={size}, sigma={sigma}, "
            f"sum={v1.sum():.6f}, max_error={max_error:.3e}"
        )

        vmax = max(v1.max(), vr.max())

        plt.subplot(len(cases), 2, 2 * index + 1) #行数 列数 位置
        plt.imshow(v1, cmap="viridis", vmin=0, vmax=vmax)
        plt.title(f"V1: {size}x{size}, sigma={sigma}")
        plt.colorbar()

        plt.subplot(len(cases), 2, 2 * index + 2)
        plt.imshow(vr, cmap="viridis", vmin=0, vmax=vmax)
        plt.title(f"OpenCV: {size}x{size}, sigma={sigma}")
        plt.colorbar()

    plt.tight_layout()
    plt.savefig("kernel_comparison.png", dpi=150)
    plt.show()