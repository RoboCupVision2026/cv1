import pathlib as pl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cbook


def zero_padding(img, pad):
    hgt, wid = img.shape
    padded = np.zeros((hgt + 2 * pad, wid + 2 * pad), dtype = img.dtype)
    padded[pad : pad + hgt, pad : pad + wid] = img
    return padded


def edge_padding(img, pad):
    hgt, wid = img.shape
    padded = zero_padding(img, pad)
    padded[pad : pad + hgt, : pad] = img[:, :1]
    padded[pad : pad + hgt, pad + wid :] = img[:, -1:]
    padded[: pad, :] = padded[pad, :]
    padded[pad + hgt :, :] = padded[pad + hgt - 1, :]
    return padded


def main():
    photo_path = cbook.get_sample_data("grace_hopper.jpg", asfileobj = False)
    photo = plt.imread(photo_path)
    img = (photo @ np.array([0.299, 0.587, 0.114])) / 255.0

    pad = 24
    fig, axes = plt.subplots(1, 3, figsize = (10, 4))
    res = [
        ("Original", img),
        ("Zero padding", zero_padding(img, pad)),
        ("Edge padding", edge_padding(img, pad)),
    ]
    for ax, (name, img) in zip(axes, res):
        ax.imshow(img, cmap="gray", vmin = 0, vmax = 1)
        ax.set_title(name)
        ax.axis("off")

    output = pl.Path(__file__).resolve().with_name("padding_results.png")
    fig.savefig(output)


if __name__ == "__main__":
    main()