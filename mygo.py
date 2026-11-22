import numpy as np


class my_GaussianBlur:
    def __init__(self, kernel_size=5, sigma=1.0):
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.kernel = self._build_kernel()

    def _build_kernel(self):
        ax = np.linspace(-(self.kernel_size - 1) / 2.,
                         (self.kernel_size - 1) / 2.,
                         self.kernel_size)
        xx, yy = np.meshgrid(ax, ax)
        kernel = np.exp(-(xx**2 + yy**2) / (2 * self.sigma**2))
        return kernel / np.sum(kernel)

    def get_kernel(self):
        return self.kernel

    def blur(self, image):
        k = self.kernel_size
        pad = k // 2
        is_color= (len(image.shape)==3)

        if is_color:
            h,w,c = image.shape
            pad_width = ((pad,pad),(pad,pad),(0,0))
            kernel = self.kernel[:,:,np.newaxis]
        else:
            h,w = image.shape
            pad_width = ((pad,pad),(pad,pad))
            kernel = self.kernel

        padded = np.pad(image, pad_width, mode='constant')
        result = np.zeros_like(image, dtype=np.float32)
        for i in range(h):
            for j in range(w):
                region = padded[i:i + k, j:j + k]
                if is_color:
                    result[i, j] = np.sum(region * kernel,axis=(0,1))
                else:
                    result[i, j] = np.sum(region * kernel)

        return np.clip(result, 0, 255).astype(np.uint8)