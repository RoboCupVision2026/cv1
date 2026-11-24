import pathlib as pl
import cv2
import matplotlib.pyplot as plt
import numpy as np


def gaussian_kernel (size, sigma):
    center = size // 2
    ker = np.zeros((size, size), dtype = float)

    for i in range(size):
        for j in range(size):
            y = i - center
            x = j - center
            ker[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return ker / ker.sum()


def main():
    settings = [(5, 1), (7, 1), (7, 2)]
    fig, axes = plt.subplots(3, 2)

    for (size, sigma), (x1, x2) in zip (settings, axes): 
        #手写的
        res1 = gaussian_kernel(size, sigma)
        #opencv的
        vec = cv2.getGaussianKernel(size, sigma)
        res2 = vec @ vec.T

        vmax = max(res1.max(), res2.max())
        for ax, ker, name in ((x1, res1, "Manual"), (x2, res2, "OpenCV")):
            ax.imshow(ker, cmap = "viridis", vmin = 0, vmax = vmax)
            ax.set_title(f"{name}: {size}x{size}, sigma={sigma:g}")
            ax.set_xticks([])
            ax.set_yticks([])

    output = pl.Path(__file__).resolve().with_name("gaussian_kernels.png")
    fig.savefig(output)


if __name__ == "__main__":
    main()