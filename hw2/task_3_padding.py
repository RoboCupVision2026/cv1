from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def zero_padding(img,pad):
    h,w=img.shape
    padded=np.zeros((h+2*pad,w+2*pad),dtype=np.uint8)
    padded[pad:pad+h,pad:pad+w]=img
    return padded
def replicate_padding(img,pad):
    h,w=img.shape
    padded=np.zeros((h+2*pad,w+2*pad),dtype=np.uint8)
    padded[pad:pad+h,pad:pad+w]=img
    for i in range(pad):
        padded[i,pad:pad+w]=img[0]
        padded[pad+h+i,pad:pad+w]=img[h-1]
    for j in range(pad):
        padded[:,j]=padded[:,pad]
        padded[:,pad+w+j]=padded[:,pad+w-1]
    return padded

if __name__=='__main__':
    image_path=Path(__file__).parent/'test_image.png'
    with Image.open(image_path) as image:
        img=np.array(image.convert('L'))
    pad=50

    zero_padded=zero_padding(img,pad)
    replicate_padded=replicate_padding(img,pad)

    plt.subplot(1,3,1)
    plt.imshow(img,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    plt.title('Original')
    plt.axis('off')
    plt.subplot(1,3,2)
    plt.imshow(zero_padded,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    plt.title(f'Zero padding, pad={pad}')
    plt.axis('off')
    plt.subplot(1,3,3)
    plt.imshow(replicate_padded,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    plt.title(f'Replicate padding, pad={pad}')
    plt.axis('off')
    plt.tight_layout()
    save_path=Path(__file__).parent/'padding_comparison.png'
    plt.savefig(save_path,dpi=300)
    plt.show()
