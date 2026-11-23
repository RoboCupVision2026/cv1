import numpy as np
import matplotlib.pyplot as plt


def zero(img, r):
    h, w = img.shape[:2]
    out = np.zeros((h + 2 * r, w + 2 * r) + img.shape[2:], img.dtype)
    out[r:r + h, r:r + w] = img
    return out


def edge(img, r):
    h, w = img.shape[:2]
    y = np.clip(np.arange(-r, h + r), 0, h - 1)
    x = np.clip(np.arange(-r, w + r), 0, w - 1)
    return img[y][:, x]


img = plt.imread("img.jpg")
figure, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img)
axes[0].set_title("original")
axes[1].imshow(zero(img, 100))
axes[1].set_title("zero padding")
axes[2].imshow(edge(img, 100))
axes[2].set_title("edge padding")
plt.tight_layout()
plt.savefig("4.png")
plt.show()
