import cv2
from My_padding import MyPadding
import matplotlib
matplotlib.use('TkAgg') 
import matplotlib.pyplot as plt

image_path = "./bird.jpg"
image=cv2.imread('bird.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 

padder = MyPadding(pad_size=75)

Model = ['Zero','Wrap around','Copy edge','Reflect across edge']

fig, axes = plt.subplots(1, 4, figsize=(12, 3), constrained_layout=True)

for i,m in enumerate(Model):
        ax = axes[i]
        padded = padder.pad(image,model=m)
        ax.imshow(padded)
        ax.set_title(f'{m}_padding', fontsize=9)
        ax.axis('off')

fig.suptitle('Padding对比', fontsize=14)
save_name = f"padding_comparison.png"
plt.savefig(save_name, dpi=300, bbox_inches='tight')
plt.show()
