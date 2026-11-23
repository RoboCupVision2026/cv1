import numpy as np
import cv2
import matplotlib.pyplot as plt


def gauss(k, s):
    r = k // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2 * s ** 2))
    return g / g.sum()


a = [(3, 1), (7, 1.5), (15, 3)]
figure, axes = plt.subplots(3, 3, figsize=(10, 10))
for i, (k, s) in enumerate(a):
    m = gauss(k, s)
    c = cv2.getGaussianKernel(k, s)
    c = c @ c.T
    axes[i][0].imshow(m, cmap="hot")
    axes[i][0].set_title(f"mine k={k} s={s}")
    axes[i][1].imshow(c, cmap="hot")
    axes[i][1].set_title(f"opencv k={k} s={s}")
    axes[i][2].imshow(np.abs(m - c), cmap="hot")
    axes[i][2].set_title(f"diff max={np.abs(m - c).max():.1e}")
plt.tight_layout()
plt.savefig("2.png")
plt.show()
