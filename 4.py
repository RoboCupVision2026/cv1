import numpy as np
import matplotlib.pyplot as plt
from skimage import data

def zero_padding(img, r):
    h, w = img.shape[:2]
    result = np.zeros(
        (h + 2 * r, w + 2 * r) + img.shape[2:],
        dtype = img.dtype
    )
    result[r : r + h, r : r + w] = img
    return result

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

img = data.cat().astype(np.uint8)

r = 30
images = [
    img,
    zero_padding(img, r),
    replicate_padding(img, r)
]

titles = [
    "Original",
    "Zero Padding",
    "Replicate Padding"
]

fig, axes = plt.subplots(1, 3, figsize = (12, 5))

for i in range(3):
    axes[i].imshow(images[i], vmin = 0, vmax = 255)
    axes[i].set_title(titles[i])
    axes[i].axis("off")

plt.tight_layout()
plt.savefig("4.png", dpi = 200)
plt.show()