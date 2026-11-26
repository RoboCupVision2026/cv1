import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif']=['SimHei','WenQuanYi Micro Hei']
plt.rcParams['axes.unicode_minus']=False

def pad_zero(img,p):
    h,w=img.shape[:2]
    if img.ndim==3:
        out=np.zeros((h+2*p,w+2*p,img.shape[2]),dtype=img.dtype)
        out[p:p+h,p:p+w]=img
    else:
        out=np.zeros((h+2*p,w+2*p),dtype=img.dtype)
        out[p:p+h,p:p+w]=img
    return out

def pad_replicate(img,p):
    # 复制边缘一圈的像素
    h,w=img.shape[:2]
    ys=np.clip(np.arange(-p,h+p),0,h-1)
    xs=np.clip(np.arange(-p,w+p),0,w-1)
    return img[ys][:,xs]

def pad_reflect(img,p):
    # 镜像，边界当成镜面翻出去
    h,w=img.shape[:2]
    ys=np.arange(-p,h+p)
    xs=np.arange(-p,w+p)
    ys=np.where(ys<0,-ys,ys)
    xs=np.where(xs<0,-xs,xs)
    ys=np.where(ys>h-1,2*(h-1)-ys,ys)
    xs=np.where(xs>w-1,2*(w-1)-xs,xs)
    return img[ys][:,xs]

def conv2d(img,kernel,mode='zero'):
    h,w=img.shape
    kh,kw=kernel.shape
    ph,pw=kh//2,kw//2
    if mode=='zero':
        padded=pad_zero(img.astype(np.float64),ph)
    elif mode=='replicate':
        padded=pad_replicate(img.astype(np.float64),ph)
    else:
        padded=pad_reflect(img.astype(np.float64),ph)
    out=np.zeros((h,w),dtype=np.float64)
    for i in range(kh):
        for j in range(kw):
            out += kernel[i][j]*padded[i:i+h,j:j+w]
    return out

img=cv2.imread('images/lena.jpg')
rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

# 先看 padding 本身长什么样
crop=rgb[180:300,180:300]
p=40

fig,axes=plt.subplots(1,4,figsize=(15,4.5))
axes[0].imshow(crop)
axes[0].set_title('原图 %dx%d'%(crop.shape[1],crop.shape[0]))
for ax,(name,out) in zip(axes[1:],[('零填充',pad_zero(crop,p)),('复制边缘',pad_replicate(crop,p)),('镜像',pad_reflect(crop,p))]):
    ax.imshow(out)
    ax.set_title('%s  pad=%d'%(name,p))
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

fig.suptitle('三种边界填充方式')
plt.tight_layout()
plt.savefig('results/padding.png',dpi=130)
plt.show()

# 再看填充方式对滤波结果边界的影响
gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
region=gray[100:400,100:400]
k=np.ones((31,31))/(31*31)

zero_blur=conv2d(region,k,'zero')
rep_blur=conv2d(region,k,'replicate')

fig,axes=plt.subplots(1,3,figsize=(13,4.8))
axes[0].imshow(np.clip(zero_blur,0,255),cmap='gray')
axes[0].set_title('零填充')
axes[1].imshow(np.clip(rep_blur,0,255),cmap='gray')
axes[1].set_title('复制边缘')
axes[2].imshow(np.abs(zero_blur-rep_blur),cmap='magma')
axes[2].set_title('两者差异')
for ax in axes:
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.savefig('results/padding_filter.png',dpi=130)
plt.show()
