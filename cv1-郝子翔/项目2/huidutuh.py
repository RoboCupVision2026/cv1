#这里是一个灰度图转换
import cv2

a = cv2.imread("input.jpg")
g = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
cv2.imwrite("gray.png", g)
