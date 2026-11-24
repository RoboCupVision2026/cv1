import cv2
import numpy as np
from gaosihe import make

a = cv2.imread("gray.png", cv2.IMREAD_GRAYSCALE)
h, w = a.shape

for k, s in [(3, 1.0), (5, 1.0), (9, 2.0)]:
    g = np.array(make(k, s))
    b = np.zeros((h - k + 1, w - k + 1))
    for y in range(h - k + 1):
        for x in range(w - k + 1):
            b[y, x] = np.sum(a[y:y+k, x:x+k] * g)
    cv2.imwrite(f"results/blur_k{k}_s{s}.png", np.rint(b).astype(np.uint8))
