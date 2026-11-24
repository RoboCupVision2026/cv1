import cv2
import numpy as np

a = cv2.imread("gray.png", cv2.IMREAD_GRAYSCALE)
h, w = a.shape
r = 30

b = np.zeros((h + 2*r, w + 2*r), dtype=np.uint8)
for y in range(h):
    for x in range(w):
        b[y+r, x+r] = a[y, x]

cv2.imwrite("zero.png", b)
