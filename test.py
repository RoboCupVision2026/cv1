import cv2
from mygo import my_GaussianBlur

image_path = "./bird.jpg"
image=cv2.imread('bird.jpg') 

my_blurrer=my_GaussianBlur(kernel_size=3, sigma=1.0)
my_gause_image=my_blurrer.blur(image)
cv_gause_image = cv2.GaussianBlur(image, (5, 5), 1.0)

cv2.imshow("my_gause_image", my_gause_image)
cv2.imshow("cv_gause_image", cv_gause_image)


cv2.waitKey(0)
cv2.destroyAllWindows()