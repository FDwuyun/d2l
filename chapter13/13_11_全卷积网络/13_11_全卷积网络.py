import torch
import torchvision
import d2l_13 as d2l
from d2l_13 import *
from torch import nn
from torch.nn import functional as F
from torchvision import models


pretrained_net = torchvision.models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
# print(list(pretrained_net.children())[-3:])

net = nn.Sequential(*list(pretrained_net.children())[:-2])

X = torch.rand(size=(1, 3, 320, 480))
# print(net(X).shape)
# torch.Size([1, 512, 10, 15])

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
# conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2, bias=False)
# W = bilinear_kernel(3, 3, 4)
# print(type(W), W.shape)
# conv_trans.weight.data.copy_(W)

# img = torchvision.transforms.ToTensor()(d2l.Image.open(d2l.script_dir + '/../img/catdog.jpg'))
# X = img.unsqueeze(0)
# print(type(X), X.shape)

# Y = conv_trans(X)
# print(type(Y), len(Y), type(Y[0]), Y[0].shape)

# out_img = Y[0].permute(1, 2, 0).detach()
# print(type(out_img), out_img.shape)

# d2l.set_figsize()
# print('input image shape:', img.permute(1, 2, 0).shape)
# d2l.plt.imshow(img.permute(1, 2, 0))
# print('output image shape:', out_img.shape)
# d2l.plt.imshow(out_img)
# # d2l.show_images([], 1, 2)
# d2l.plt.savefig(d2l.script_dir + "/转置卷积放大2倍.png")


# 全卷积网络用双线性插值的上采样初始化转置卷积层。对于卷积层，我们使用Xavier初始化参数。
W = bilinear_kernel(num_classes, num_classes, 64)
nn.init.xavier_uniform_(net.final_conv.weight)
net.transpose_conv.weight.data.copy_(W)

ps = {
    "net": "models_resnet18[:-2] + final_conv + transpose_conv",
    "weights": "",
    "num_gpus": 5,
    "num_epochs": 1,
    "batch_size": 32,
    "lr": 0.0005,
    "weight_decay": 0.001,
    "description": ""
}

# 读取数据集
batch_size, crop_size = ps["batch_size"], (320, 480)
# train_iter, test_iter = d2l.load_data_voc(batch_size, crop_size)

# 训练
def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)
num_epochs, lr, wd, devices = ps["num_epochs"], ps["lr"], ps["weight_decay"], d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)

# # log日志记录
# log_dir = os.path.join(script_dir, "logs", f"全卷积网络_{get_datetime()}")
# # 实例化TensorBoard SummaryWriter对象
# tb_writer = SummaryWriter(log_dir = log_dir)
# # 实例化Logger对象
# sys.stdout = Logger(log_dir + f"/output_{get_datetime()}.txt")

# if __name__ == '__main__':
#     print(ps)
#     d2l.load_model_param(net, f"{d2l.script_dir}" + "/FullyConvNet_params.pth")
#     train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, 2)
#     plt.savefig(script_dir + f"/全卷积网络.png")
#     d2l.save_model_param(net, f"{d2l.script_dir}" + "/FullyConvNet_params.pth")

d2l.load_model_param(net, f"{d2l.script_dir}" + "/FullyConvNet_params.pth")
net = net.to(devices[0])
"""问题在此，仔细分析！！！"""
net.eval()

# 预测
def predict(img):
    """在预测时，我们需要将输入图像在各个通道做标准化，并转成卷积神经网络所需要的四维输入格式。"""
    transform = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    X = transform(img.float() / 255).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    """为了可视化预测的类别给每个像素，我们将预测类别映射回它们在数据集中的标注颜色。"""
    colormap = torch.tensor(d2l.VOC_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

voc_dir = os.path.join(script_dir, "..", "..", "data/VOCdevkit/VOC2012")
test_images, test_labels = d2l.read_voc_images(voc_dir, False)
n, imgs = 4, []
for i in range(n):
    # crop_rect = (0, 0, 320, 480)
    X, y = voc_rand_crop(test_images[i], test_labels[i], height=320, width=480)
    pred = label2image(predict(X))
    imgs += [X.permute(1,2,0), pred.cpu(), y.permute(1,2,0)]
    
d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=4)
plt.savefig(script_dir + f"/预测.png")
print("预测完毕！")
