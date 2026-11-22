import cv2
from mygo import my_GaussianBlur
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

image_path = "./bird.jpg"
m_image = cv2.imread(image_path)
img_rgb = cv2.cvtColor(m_image, cv2.COLOR_BGR2RGB)


ksize = [3,7,11,21]
sigma = [1.0,2.0,4.0,12.0]

fig, axes = plt.subplots(4, 4, figsize=(12, 12), constrained_layout=True)

for i, k in enumerate(ksize):
    for j, s in enumerate(sigma):
        ax = axes[i, j]
        blurrer = my_GaussianBlur(kernel_size=k,sigma=s)
        blurred = blurrer.blur(img_rgb)
        ax.imshow(blurred)
        ax.set_title(f'ksize={k}, sigma={s}', fontsize=9)
        ax.axis('off')

fig.suptitle('高斯核 大小*方差 对比', fontsize=14)
save_name = f"multiple_comparison.png"
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()
