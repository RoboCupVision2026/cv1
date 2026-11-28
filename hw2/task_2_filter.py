from task_1_kernel import gaussian_kernel
from pathlib import Path
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def gaussian_blur(img,size,sigma):
    h=np.array(gaussian_kernel(size,sigma))
    pad=size//2
    padded=np.pad(img,((pad,pad),(pad,pad)),mode='reflect')
    filtered=np.zeros(img.shape)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            window=padded[i:i+size,j:j+size]
            filtered[i,j]=np.sum(window*h)
    return np.rint(filtered).astype(np.uint8)

if __name__=='__main__':
    image_path=Path(__file__).parent/'test_image.png'
    with Image.open(image_path) as image:
        img=np.array(image.convert('L'))
    confs=[(11,3),(21,3),(21,6)]

    plt.subplot(2,2,1)
    plt.imshow(img,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
    plt.title('Original')
    plt.axis('off')

    for i in enumerate(confs,2):
        filtered=gaussian_blur(img,i[1][0],i[1][1])
        plt.subplot(2,2,i[0])
        plt.imshow(filtered,cmap='gray',vmin=0,vmax=255,interpolation='nearest')
        plt.title(f'{i[1][0]}x{i[1][0]}, sigma={i[1][1]}')
        plt.axis('off')

    plt.tight_layout()
    save_path=Path(__file__).parent/'filtered_image_comparison.png'
    plt.savefig(save_path,dpi=300)
    plt.show()