
import torch
from torchvision.models import vgg11
import matplotlib
matplotlib.use("Agg")  # 这一句一定要放在下面这句的前面
from matplotlib import pyplot as plt
import os
from PIL import Image
import numpy as np

script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "CNN可视化")
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

model = vgg11(pretrained=True)
print(dict(model.features.named_children()))

conv1 = dict(model.features.named_children())["3"]
kernel_set = conv1.weight.detach()
num = len(conv1.weight.detach())
print(kernel_set.shape)
for i in range(0, num):
    i_kernel = kernel_set[i]
    plt.figure(figsize=(20, 17))
    if (len(i_kernel)) > 1:
        for idx, filer in enumerate(i_kernel):
            plt.subplot(9, 9, idx + 1)
            plt.axis("off")
            plt.imshow(filer[:, :].detach(), cmap="bwr")
        
        plt.savefig(data_dir + "/table" + str(i) +  ".png")

