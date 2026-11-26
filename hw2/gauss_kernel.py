import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei','WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus']=False

def gaussian_kernel(ksize,sigma):
    k=np.zeros((ksize,ksize),dtype=np.float64)
    c=ksize//2
    for i in range(ksize):
        for j in range(ksize):
            d=(i-c)**2+(j-c)**2
            k[i][j]=np.exp(-d/(2*sigma*sigma))
    # 不归一化的话滤波以后整张图会变暗
    k=k/k.sum()
    return k

def cv_kernel(ksize,sigma):
    g=cv2.getGaussianKernel(ksize,sigma)
    return np.outer(g,g)

cases=[(3,1.0),(7,2.0),(15,5.0)]

fig,axes=plt.subplots(len(cases),2,figsize=(7,10))
for row,(ksize,sigma) in enumerate(cases):
    mine=gaussian_kernel(ksize,sigma)
    ref=cv_kernel(ksize,sigma)
    diff=np.abs(mine-ref).max()
    print('ksize = %d, sigma = %.1f, 最大误差 = %.3e'%(ksize,sigma,diff))

    for col,(k,name) in enumerate([(mine,'手动实现'),(ref,'cv2.getGaussianKernel')]):
        ax=axes[row][col]
        im=ax.imshow(k,cmap='viridis')
        ax.set_title('%s %dx%d sigma=%.1f'%(name,ksize,ksize,sigma))
        ax.set_xticks([])
        ax.set_yticks([])
        plt.colorbar(im,ax=ax,fraction=0.046)

fig.suptitle('二维高斯滤波核')
plt.tight_layout()
plt.savefig('results/gaussian_kernel.png',dpi=150)
plt.show()
