import os
import torch
import torchvision
import d2l_13 as d2l
from d2l_13 import *


# d2l.DATA_HUB['voc2012'] = (d2l.DATA_URL + 'VOCtrainval_11-May-2012.tar',
#                            '4e443f8a2eca6b1dac8a6c57641b67dd40621a49')

# voc_dir = d2l.download_extract('voc2012', 'VOCdevkit/VOC2012')
# print(voc_dir)
voc_dir = os.path.join(d2l.script_dir, "..", "..", "data/VOCdevkit/VOC2012")


# train_features, train_labels = d2l.read_voc_images(voc_dir, True)

# n = 5
# imgs = train_features[0:n] + train_labels[0:n]
# print(type(imgs), len(imgs), type(imgs[0]), imgs[0].shape, imgs[1].shape)
# imgs = [img.permute(1,2,0) for img in imgs]
# print(type(imgs), type(imgs[0]), imgs[0].shape, imgs[1].shape)
# d2l.show_images(imgs, 2, n)
# d2l.plt.savefig(f"{d2l.script_dir}" + "/前5个输入图像及其标签.png")

# y = voc_label_indices(train_labels[0], voc_colormap2label())
# print(y[105:115, 130:140], VOC_CLASSES[1])

# imgs = []
# for _ in range(n):
#     imgs += voc_rand_crop(train_features[0], train_labels[0], 200, 300)
# imgs = [img.permute(1, 2, 0) for img in imgs]
# d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
# d2l.plt.savefig(f"{d2l.script_dir}" + "/图像裁剪为固定尺寸.png")

crop_size = (320, 480)
voc_train = VOCSegDataset(True, crop_size, voc_dir)
voc_test = VOCSegDataset(False, crop_size, voc_dir)

batch_size = 64
train_iter = torch.utils.data.DataLoader(voc_train, batch_size, shuffle=True,
                                    drop_last=True,
                                    num_workers=d2l.get_dataloader_workers())
for X, Y in train_iter:
    print(X.shape)
    print(Y.shape)
    break