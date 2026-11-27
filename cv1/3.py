import numpy as np
import cv2

def zero_padding(image, p):
    h, w = image.shape[:2]
    output = np.zeros((h + 2*p, w + 2*p, 3), dtype=image.dtype)
    output[p:p+h, p:p+w] = image
    return output

def copy_padding(image, p):
    h, w = image.shape[:2]
    output = np.zeros((h + 2*p, w + 2*p, 3), dtype=image.dtype)
    output[p:p+h, p:p+w] = image
    output[:p, p:p+w] = image[0:1, :, :]
    output[p+h:, p:p+w] = image[h-1:h, :, :]
    output[:, :p] = output[:, p:p+1]
    output[:, p+w:] = output[:, p+w-1:p+w]
    return output

image = cv2.imread('/home/lihaoxuan/robocup/cv1/ayaka.png', cv2.IMREAD_COLOR)
image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
img_0 = zero_padding(image, 100)
img_copy = copy_padding(image, 100)
result = np.hstack((img_0, img_copy))
cv2.imwrite('/home/lihaoxuan/robocup/cv1/task3.png', result)
