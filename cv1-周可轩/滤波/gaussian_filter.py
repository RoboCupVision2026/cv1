import pathlib as pl
import sys
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook

def gaussian_kernel (size, sigma):
    center = size // 2
    ker = np.zeros((size, size), dtype = float)

    for i in range(size):
        for j in range(size):
            y = i - center
            x = j - center
            ker[i, j] = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    return ker / ker.sum()

def filter(img, ker):
    hgt, wid = img.shape
    size = ker.shape[0]
    pad = size // 2

    padded = np.zeros((hgt + 2 * pad, wid + 2 * pad), dtype = float)
    padded[pad : pad + hgt, pad : pad + wid] = img
    res = np.zeros((hgt, wid), dtype=float)

    for i in range(hgt):
        for j in range(wid):
            now = padded[i : i + size, j : j + size]
            res[i, j] = (now * ker).sum()

    return res


def main():
    #用的matplotlib自带图片
    photo_path = cbook.get_sample_data("grace_hopper.jpg", asfileobj = False)
    photo = plt.imread(photo_path)
    img = (photo @ np.array([0.299, 0.587, 0.114])) / 255.0

    settings = [(5, 1), (7, 1), (7, 2)]
    fig, axes = plt.subplots(2, 2, figsize = (8, 8))
    axes = axes.ravel()
    axes[0].imshow(img, cmap = "gray", vmin = 0, vmax = 1)
    axes[0].set_title("Original")

    for ax, (size, sigma) in zip(axes[1:], settings):
        ker = gaussian_kernel(size, sigma)
        fil = filter(img, ker)
        ax.imshow(fil, cmap = "gray", vmin = 0, vmax = 1)
        ax.set_title(f"{size}x{size}, sigma = {sigma}")

    for ax in axes:
        ax.axis("off")

    output = pl.Path(__file__).resolve().with_name("gaussian_filter_results.png")
    fig.savefig(output)


if __name__ == "__main__":
    main()
