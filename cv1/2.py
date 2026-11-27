import numpy as np
import cv2

def kernel(size, sigma):
    r = size // 2
    x = np.arange(-r, r+1)
    y = np.arange(-r, r+1)
    X, Y = np.meshgrid(x, y)
    kernel = np.exp(-(X**2 + Y**2) / (2 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel

def filter(image, size, sigma):
    ker = kernel(size, sigma)
    r = size // 2

    padded = np.pad(image, ((r, r), (r, r), (0, 0)), mode='reflect')
    
    height, width = image.shape[:2]
    output = np.zeros((height, width, 3), dtype=np.float64)
    for i in range(height):
        for j in range(width):
            for k in range(3):
                region = padded[i:i+size, j:j+size, k]
                output[i, j, k] = np.sum(region * ker)
    output = np.clip(output, 0, 255).astype(np.uint8)
    return output

IMAGE = '/home/lihaoxuan/robocup/cv1/robin.png'
image = cv2.imread(IMAGE, cv2.IMREAD_COLOR)
image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25) #没招了，图片有一点点大:)
filtered0 = filter(image, 5, 1.0)
filtered1 = filter(image, 15, 3.0)
filtered2 = filter(image, 25, 5.0)
result = np.hstack((image, filtered0, filtered1, filtered2))
cv2.imwrite('/home/lihaoxuan/robocup/cv1/task2.png', result)
