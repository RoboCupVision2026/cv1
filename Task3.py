import numpy as np
import cv2
import matplotlib.pyplot as plt

img_path=input("请输入想要处理图片的路径：")
img=cv2.imread(img_path)
if img is None:
    raise FileNotFoundError("图片读取失败，请检查路径是否正确")
img_rgb=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
img=img.astype(float)
h,w,cha=img.shape
print("请输入高斯滤波核的大小（奇数）和sigma，空格隔开：")
size_str,sigma_str=input().split()
size=int(size_str)
#滤波不能用库函数，生成滤波核应该可以罢:)
kernel_ver=cv2.getGaussianKernel(size,float(sigma_str))
kernel=kernel_ver@kernel_ver.T

names=["Zero","Replicate"]
img_finals=[]
p=(size-1)//2
img_final1=np.zeros((h+2*p,w+2*p,cha))
img_final1[p:p+h,p:p+w,0:cha]=img
#Zero Padding
img_finals.append(img_final1)
#Replicate Padding
img_final2=img_final1.copy()
for i in range(p):
    img_final2[i,:]=img_final2[p,:]
    img_final2[-i-1,:]=img_final2[-p-1,:]
for i in range(p):
    img_final2[:,i]=img_final2[:,p]
    img_final2[:,-i-1]=img_final2[:,-p-1]
img_finals.append(img_final2)

outs=[]
for i in range(2):
    print(f"开始对{names[i]} Padding后的原图进行滤波")
    hf,wf,chanel=img_finals[i].shape
    output=np.zeros((hf-size+1,wf-size+1,chanel))
    for j in range(hf-size+1):
        for k in range(wf-size+1):
            for c in range(chanel):
                output[j,k,c] = np.sum(img_finals[i][j:j+size,k:k+size,c]*kernel)
    output=np.clip(output,0,255).astype(np.uint8)
    outs.append(cv2.cvtColor(output,cv2.COLOR_BGR2RGB))
    print(f"{names[i]} Padding后的原图滤波完成")

plt.figure(figsize=(14,18))
plt.subplot(3,2,(1,2))
plt.imshow(img_rgb)
plt.title("原图")
plt.axis("off")
for i in range(2):
    plt.subplot(3,2,i+3)
    plt.imshow(cv2.cvtColor(np.clip(img_finals[i],0,255).astype(np.uint8),cv2.COLOR_BGR2RGB))
    plt.title(f"{names[i]} Padding后的原图")
    plt.axis("off")
for i in range(2):
    plt.subplot(3,2,i+5)
    plt.imshow(outs[i])
    plt.title(f"{names[i]} Padding后滤波(Size={size},sigma={float(sigma_str)})完成的图片")
    plt.axis("off")
plt.tight_layout()
plt.show()
