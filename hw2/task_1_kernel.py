import math
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def gaussian_func(sigma,x,y):
    return math.exp(-((x**2+y**2)/(2*sigma**2)))/(2*math.pi*sigma**2)
def gaussian_kernel(size,sigma):
    center=(size-1)/2
    kernel=[]
    total=0
    for i in range(size):
        weights=[]
        for j in range(size):
            x=j-center
            y=i-center
            weight=gaussian_func(sigma,x,y)
            weights.append(weight)
            total+=weight
        kernel.append(weights)
    for i in range(size):
        for j in range(size):
            kernel[i][j]/=total
    return kernel

if __name__=='__main__':
    import cv2
    confs=[(5,1),(15,3),(15,5)]
    
    for i in enumerate(confs):
        h=np.array(gaussian_kernel(i[1][0],i[1][1]))
        g=cv2.getGaussianKernel(i[1][0],i[1][1])
        g=g@g.T
        max_weight=max(h.max(),g.max())
        
        plt.subplot(3,2,i[0]*2+1)
        plt.imshow(h,cmap='Blues_r',vmin=0,vmax=max_weight,interpolation='nearest')
        plt.title(f'Handwritten: {i[1][0]}x{i[1][0]}, sigma={i[1][1]}')
        plt.subplot(3,2,i[0]*2+2)
        plt.imshow(g,cmap='Blues_r',vmin=0,vmax=max_weight,interpolation='nearest')
        plt.title(f'OpenCV: {i[1][0]}x{i[1][0]}, sigma={i[1][1]}')

    plt.tight_layout()
    save_path=Path(__file__).parent/'kernel_comparison.png'
    plt.savefig(save_path,dpi=300)
    plt.show()