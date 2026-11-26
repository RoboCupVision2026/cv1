import time
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
    k=k/k.sum()
    return k

def conv2d(img,kernel):
    h,w=img.shape
    kh,kw=kernel.shape
    ph,pw=kh//2,kw//2
    padded=np.zeros((h+2*ph,w+2*pw),dtype=np.float64)
    padded[ph:ph+h,pw:pw+w]=img
    # 卷积就是核里每个位置乘上一张平移过的图，全部加起来
    out=np.zeros((h,w),dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i][j]*padded[i:i+h,j:j+w]
    return out

def gaussian_blur(img,ksize,sigma):
    k=gaussian_kernel(ksize,sigma)
    if img.ndim==2:
        return conv2d(img.astype(np.float64),k)
    out=np.zeros(img.shape,dtype=np.float64)
    for ch in range(img.shape[2]):
        out[:,:,ch]=conv2d(img[:,:,ch].astype(np.float64),k)
    return out

img=cv2.imread('images/lena.jpg')
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
print('图像大小',rgb.shape)

params=[(5,1.0),(15,3.0),(31,8.0)]

fig,axes=plt.subplots(2,4,figsize=(15,8.6))
axes[0][0].imshow(rgb)
axes[0][0].set_title('原图')
axes[1][0].imshow(rgb[200:340,180:320])
axes[1][0].set_title('原图局部')

for col,(ksize,sigma) in enumerate(params):
    t=time.time()
    out=gaussian_blur(rgb,ksize,sigma)
    out=np.clip(out,0,255).astype(np.uint8)
    print('核 %dx%d sigma=%.1f 用时 %.2f 秒'%(ksize,ksize,sigma,time.time()-t))

    axes[0][col+1].imshow(out)
    axes[0][col+1].set_title('%dx%d sigma=%.1f'%(ksize,ksize,sigma))
    axes[1][col+1].imshow(out[200:340,180:320])
    axes[1][col+1].set_title('局部')

for ax in axes.flatten():
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('手动实现的高斯模糊')
plt.tight_layout(rect=(0,0,1,0.96))
plt.savefig('results/blur_compare.png',dpi=130)
plt.show()
