import torch
import d2l_13 as d2l
from d2l_13 import *
import torchvision
from torch import nn
from torch.nn import functional as F

pretrained_net = torchvision.models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
net = nn.Sequential(*list(pretrained_net.children())[:-2])
batch_size, crop_size = 32, (320, 480)
train_iter, val_iter = d2l.load_data_voc(batch_size, crop_size)

num_classes = 21
net.add_module('final_conv', nn.Conv2d(512, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                    kernel_size=64, padding=16, stride=32))
def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, weight_decay, devices = 2, 1e-3, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(params=net.parameters(), lr=lr, weight_decay=weight_decay)

d2l.load_model_param(net, f"{d2l.script_dir}" + "/FullyConvNet_params.pth")
d2l.train_ch13(net, train_iter, val_iter, loss, trainer, num_epochs, 1)
d2l.save_model_param(net, f"{d2l.script_dir}" + "/FullyConvNet_params1.pth")

# val_acc = d2l.evaluate_accuracy_gpu(net, val_iter, devices[0])
# print(val_acc)
# exit()
def predict(net, img):
    X = val_iter.dataset.normalize_image(img).unsqueeze(0)
    preds = net(X.to(devices[0])).argmax(dim=1)
    return preds.reshape(preds.shape[1], preds.shape[2])

def label2image(pred):
    colormap = torch.tensor(d2l.VOC_COLORMAP)
    pred = pred.long()
    return colormap[pred, :]

voc_dir = os.path.join(script_dir, "..", "..", "data/VOCdevkit/VOC2012")
test_images, test_labels = d2l.read_voc_images(voc_dir, is_train=False)
n, images = 4, []
for i in range(n):
    crop_rect = (0, 0, 320, 480)
    crop_img = torchvision.transforms.functional.crop(test_images[i], *crop_rect)
    crop_label = torchvision.transforms.functional.crop(test_labels[i], *crop_rect)
    pred_image = label2image(predict(net, crop_img))
    images += [crop_img.permute(1, 2, 0), pred_image.cpu(), crop_label.permute(1, 2, 0)]
d2l.show_images(images[::3] + images[1::3] + images[2::3], 3, n, scale=2)
d2l.plt.savefig(d2l.script_dir + f"/预测.png")
print("预测完毕！")
