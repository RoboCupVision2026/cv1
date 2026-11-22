import cv2
import numpy as np
import matplotlib.pyplot as plt
def generate_gaussian_kernel_2d(size, sigma):
    size = int(size) //2
    x,y = np.mgrid[-size:size+1, -size:size+1]
    normal = 1/ (2.0 * np.pi * sigma**2)
    kernel = np.exp(-((x**2 + y**2) / (2.0*sigma**2))) * normal
    kernel = kernel / np.sum(kernel)
    return kernel
params =[(3,0.8),(5,1.0),(7,1.5)]
for i,(size,sigma) in enumerate(params):
    my_kernel = generate_gaussian_kernel_2d(size,sigma)
    opencv_kernel_1d = cv2.getGaussianKernel(size,sigma)
    opencv_kernel_2d = np.dot(opencv_kernel_1d,opencv_kernel_1d.T)
    plt.figure(figsize=(10,10))
    plt.subplot(1,2,1)
    plt.imshow(my_kernel)
    plt.title(f"MINE(size={size})")
    plt.colorbar()
    plt.subplot(1,2,2)
    plt.imshow(opencv_kernel_2d)
    plt.title(f"OpenCV(size={size})")
    plt.colorbar()
    plt.savefig(f"第{i+1}组.png")
print("done1")


def my_convolved2d(image, kernel):
    k = kernel.shape[0] // 2
    padded_image = np.pad(image, ((k, k), (k, k)), mode='constant', constant_values=0)
    convolved_image = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            convolved_image[i, j] = np.sum(padded_image[i:i + kernel.shape[0], j:j + kernel.shape[1]] * kernel)
    return convolved_image
img = cv2.imread("1.jpg",0)
img = cv2.resize(img,(200,150))
plt.figure(figsize=(10,10))
plt.subplot(1,4,1)
plt.imshow(img,cmap='gray')
plt.title("Original Image")
for i,(size,sigma) in enumerate(params):
    kernel = generate_gaussian_kernel_2d(size,sigma)
    blurred_image = my_convolved2d(img, kernel)
    plt.subplot(1,4,i+2)
    plt.imshow(blurred_image,cmap='gray')
    plt.title(f"Blurred Image(size={size})")
plt.tight_layout()
plt.savefig("blurred_images.png",dpi=150)
plt.show()
print("done2")

img = cv2.imread("1.jpg",0)
img = cv2.resize(img,(200,150))
k=30
padded_edge = np.pad(img,((k,k),(k,k)),mode='edge')
padded_reflect = np.pad(img,((k,k),(k,k)),mode='reflect')
plt.figure(figsize=(10,10))
plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.title("Original")
plt.subplot(1,3,2)
plt.imshow(padded_edge,cmap='gray')
plt.title("Edge")
plt.subplot(1,3,3)
plt.imshow(padded_reflect,cmap='gray')
plt.title("Reflect")
plt.tight_layout()
plt.savefig("padding_images.png",dpi=150)
plt.show()
print("done3")
