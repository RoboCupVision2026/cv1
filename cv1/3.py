import numpy as np
import matplotlib.pyplot as plt


def gauss(k, s):
    r = k // 2
    y, x = np.mgrid[-r:r + 1, -r:r + 1]
    g = np.exp(-(x ** 2 + y ** 2) / (2 * s ** 2))
    return g / g.sum()


def conv(img, m):
    k = m.shape[0]
    r = k // 2
    h, w = img.shape
    p = np.pad(img, r, mode="edge")
    out = np.zeros((h, w))
    for i in range(k):
        for j in range(k):
            out += m[i, j] * p[i:i + h, j:j + w]
    return out


img = plt.imread("img.jpg").mean(axis=2)
a = [(3, 1), (7, 1.5), (15, 3)]
figure, axes = plt.subplots(2, 2, figsize=(12, 9))
axes[0][0].imshow(img, cmap="gray")
axes[0][0].set_title("original")
for n, (k, s) in enumerate(a, 1):
    axes[n // 2][n % 2].imshow(conv(img, gauss(k, s)), cmap="gray")
    axes[n // 2][n % 2].set_title(f"k={k} s={s}")
plt.tight_layout()
plt.savefig("3.png")
plt.show()
