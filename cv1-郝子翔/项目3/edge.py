import cv2
import numpy as np

a = cv2.imread("gray.png", cv2.IMREAD_GRAYSCALE)
h, w = a.shape
r = 30

b = np.zeros((h + 2*r, w + 2*r), dtype=np.uint8)
for y in range(h + 2*r):
    for x in range(w + 2*r):
        yy = min(max(y-r, 0), h-1)
        xx = min(max(x-r, 0), w-1)
        b[y, x] = a[yy, xx]

cv2.imwrite("edge.png", b)
