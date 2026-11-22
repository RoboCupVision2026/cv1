import numpy as np

class MyPadding:
    def __init__(self, pad_size):
        self.pad_size = pad_size

    def pad(self, image, model='zero'):
        p = self.pad_size
        is_color = (len(image.shape) == 3)

        if is_color:
            h, w, c = image.shape
            new_shape = (h + 2 * p, w + 2 * p, c)
        else:
            h, w = image.shape
            new_shape = (h + 2 * p, w + 2 * p)

        new_img = np.zeros(new_shape, dtype=np.uint8)
        new_img[p:p + h, p:p + w] = image

        if model == 'zero':
            return new_img

        elif model == 'Wrap around':
            new_img[0:p, p:p + w] = image[h - p:h, :]
            new_img[p + h:2 * p + h, p:p + w] = image[0:p, :]
            new_img[p:p + h, 0:p] = image[:, w - p:w]
            new_img[p:p + h, p + w:2 * p + w] = image[:, 0:p]

            new_img[0:p, 0:p] = image[h - p:h, w - p:w]
            new_img[0:p, p + w:2 * p + w] = image[h - p:h, 0:p]
            new_img[p + h:2 * p + h, 0:p] = image[0:p, w - p:w]
            new_img[p + h:2 * p + h, p + w:2 * p + w] = image[0:p, 0:p]
            return new_img

        elif model == 'Copy edge':
            new_img[0:p, p:p + w] = image[0:1, :]
            new_img[p + h:2 * p + h, p:p + w] = image[h - 1:h, :]
            new_img[p:p + h, 0:p] = image[:, 0:1]
            new_img[p:p + h, p + w:2 * p + w] = image[:, w - 1:w]

            new_img[0:p, 0:p] = image[0, 0]
            new_img[0:p, p + w:2 * p + w] = image[0, w - 1]
            new_img[p + h:2 * p + h, 0:p] = image[h - 1, 0]
            new_img[p + h:2 * p + h, p + w:2 * p + w] = image[h - 1, w - 1]
            return new_img

        elif model == 'Reflect across edge':
            new_img[0:p, p:p + w] = image[p:0:-1, :]
            new_img[p + h:2 * p + h, p:p + w] = image[h - 2:h - 2 - p:-1, :]
            new_img[p:p + h, 0:p] = image[:, p:0:-1]
            new_img[p:p + h, p + w:2 * p + w] = image[:, w - 2:w - 2 - p:-1]

            new_img[0:p, 0:p] = image[p:0:-1, p:0:-1] 
            new_img[0:p, p + w:2 * p + w] = image[p:0:-1, w - 2:w - 2 - p:-1]
            new_img[p + h:2 * p + h, 0:p] = image[h - 2:h - 2 - p:-1, p:0:-1]
            new_img[p + h:2 * p + h, p + w:2 * p + w] = image[h - 2:h - 2 - p:-1, w - 2:w - 2 - p:-1]
            
            return new_img

        return new_img

