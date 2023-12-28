import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models,transforms,datasets
from torch.utils import data

import d2l_13 as d2l

def class_predictor(num_inputs, num_anchors, num_classes):
    return nn.Conv2d(
        num_inputs, num_anchors * (num_classes + 1),
        kernel_size=3, padding=1
    )

def bbox_predictor(num_inputs, num_anchors):
    """为每个锚框预测4个偏移量，偏移量是真实边界框相对于锚框的偏移量"""
    return nn.Conv2d(num_inputs, num_anchors * 4, kernel_size=3, padding=1)

def forward(x, block):
    return block(x)


def flatten_pred(pred):
    return torch.flatten(pred.permute(0, 2, 3, 1), start_dim=1)


def concat_preds(preds):
    return torch.cat([flatten_pred(p) for p in preds], dim=1)


def down_sample_blk(in_channels, out_channels):
    blk = []
    for _ in range(2):
        blk.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        blk.append(nn.BatchNorm2d(out_channels))
        blk.append(nn.ReLU())
        in_channels = out_channels
    blk.append(nn.MaxPool2d(kernel_size=2)) # 步长默认值是kernel_size,(256-2+0+2)/2
    return nn.Sequential(*blk)


def base_net():
    blk = []
    num_filters = [3, 16, 32, 64]
    for i in range(len(num_filters) - 1):
        blk.append(down_sample_blk(num_filters[i], num_filters[i+1]))
    return nn.Sequential(*blk)


def get_blk(i):
    if i == 0:
        blk = base_net()
    elif i == 1:
        blk = down_sample_blk(64, 128)
    elif i == 4:
        blk = nn.AdaptiveMaxPool2d((1, 1))
    else:
        blk = down_sample_blk(128, 128)
    return blk

def blk_forward(X, blk, size, ratio, cls_predictor, bbox_predictor):
    Y = blk(X)
    anchors = d2l.multibox_prior(Y, sizes=size, ratios=ratio)
    class_preds = cls_predictor(Y)
    bbox_preds = bbox_predictor(Y)
    return (Y, anchors, class_preds, bbox_preds)

class TinySSD(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(TinySSD, self).__init__(**kwargs)
        self.num_classes = num_classes
        idx_to_in_channels = [64, 128, 128, 128, 128]
        for i in range(5):
            # 即赋值语句self.blk_i = get_blk(i)
            # 设置对象的属性值。在这里，self 是模型实例，f'blk_{k}' 是属性名，get_blk(i) 是属性值。
            setattr(self, f'blk_{i}', get_blk(i))
            setattr(self, f'cls_{i}', class_predictor(idx_to_in_channels[i], num_anchors, num_classes))
            setattr(self, f'bbox_{i}', bbox_predictor(idx_to_in_channels[i], num_anchors))

    def forward(self, X):
        anchors, cls_preds, bbox_preds = [None] * 5, [None] * 5, [None] * 5
        for i in range(5):
            # getattr(self, 'blk_%d'%i) 即访问self.blk_i
            X, anchors[i], cls_preds[i], bbox_preds[i] = blk_forward(
                X,
                getattr(self, f'blk_{i}'), sizes[i], ratios[i],
                getattr(self, f'cls_{i}'), getattr(self, f'bbox_{i}')
            )
        # print(type(anchors), len(anchors), anchors[0].shape, anchors[1].shape, anchors[2].shape, anchors[3].shape, anchors[4].shape)
        anchors = torch.cat(anchors, dim=1)
        cls_preds = concat_preds(cls_preds)
        cls_preds = cls_preds.reshape(cls_preds.shape[0], -1, self.num_classes+1)
        bbox_preds = concat_preds(bbox_preds)
        return anchors, cls_preds, bbox_preds

def calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels, bbox_masks):
    """
    目标检测有两种类型的损失。 第一种有关锚框类别的损失：我们可以简单地复用之前图像分类问题里一直使用的交叉熵损失函数来计算； 第二种有关正类锚框偏移量的损失：预测偏移量是一个回归问题。 但是，对于这个回归问题，我们在这里不使用 3.1.3节中描述的平方损失，而是使用
    范数损失，即预测值和真实值之差的绝对值。
    掩码变量bbox_masks令负类锚框和填充锚框不参与损失的计算。
    """
    batch_size, num_classes = cls_preds.shape[0], cls_preds.shape[2]
    cls = cls_loss(cls_preds.reshape(-1, num_classes), cls_labels.reshape(-1)).reshape(batch_size, -1).mean(dim=1)
    bbox = bbox_loss(bbox_preds * bbox_masks, bbox_labels * bbox_masks).mean(dim=1)
    return cls + bbox

def cls_eval(cls_preds, cls_labels):
    """由于类别预测结果放在最后一维，argmax需要指定最后一维。"""
    return float((cls_preds.argmax(dim=-1).type(cls_labels.dtype) == cls_labels).sum())

def bbox_eval(bbox_preds, bbox_labels, bbox_masks):
    return float((torch.abs((bbox_labels - bbox_preds) * bbox_masks)).sum())

# main

# Y1 = forward(torch.zeros((2, 8, 20, 20)), class_predictor(8, 5, 10))
# Y2 = forward(torch.zeros((2, 16, 10, 10)), class_predictor(16, 3, 10))
# print(Y1.shape, Y2.shape)

# print(concat_preds([Y1, Y2]).shape)

# print(forward(torch.zeros((2, 3, 20, 20)), down_sample_blk(3, 10)).shape)

# print(forward(torch.zeros((2, 3, 256, 256)), base_net()).shape)

sizes = [
    [0.2, 0.272],
    [0.37, 0.447],
    [0.54, 0.619],
    [0.71, 0.79],
    [0.88, 0.961]
]
ratios = [[1, 2, 0.5]] * 5
num_anchors = len(sizes[0]) + len(ratios[0]) - 1

# net = TinySSD(num_classes=1)
# X = torch.zeros((32, 3, 256, 256))
# d2l.summary(net, input_data = X)
# anchors, cls_preds, bbox_preds = net(X)
# print('output anchors:', anchors.shape)
# print('output class preds:', cls_preds.shape)
# print('output bbox preds:', bbox_preds.shape)

batch_size = 32
# train_iter, _ = d2l.load_data_bananas(batch_size)
# for X, Y in train_iter:
#     print(X.shape)
#     print(Y.shape)
#     break
# exit(0)

device, net = d2l.try_gpu(), TinySSD(num_classes=1)
trainer = torch.optim.SGD(net.parameters(), lr=0.2, weight_decay=5e-4)

cls_loss = nn.CrossEntropyLoss(reduction='none')
bbox_loss = nn.L1Loss(reduction='none')

num_epochs, timer = 20, d2l.Timer()
animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs], legend=['class error', 'bbox mae'])

net = net.to(device)
# d2l.load_model_param(net, f"{d2l.script_dir}" + "/SSD_params.pth")
# for epoch in range(num_epochs):
#     # 训练精确度的和，训练精确度的和中的示例数
#     # 绝对误差的和，绝对误差的和中的示例数
#     metric = d2l.Accumulator(4)
#     net.train()
#     for features, target in train_iter:
#         timer.start()
#         trainer.zero_grad()
#         X, Y = features.to(device), target.to(device)
#         # 生成多尺度的锚框，为每个锚框预测类别和偏移量
#         anchors, cls_preds, bbox_preds = net(X)
#         # 为每个锚框标注类别和偏移量
#         bbox_labels, bbox_masks, cls_labels = d2l.multibox_target(anchors, Y)
#         # 根据类别和偏移量的预测和标注值计算损失函数
#         l = calc_loss(cls_preds, cls_labels, bbox_preds, bbox_labels,
#                       bbox_masks)
#         l.mean().backward()
#         trainer.step()
#         metric.add(cls_eval(cls_preds, cls_labels), cls_labels.numel(),
#                    bbox_eval(bbox_preds, bbox_labels, bbox_masks),
#                    bbox_labels.numel())
#     cls_err, bbox_mae = 1 - metric[0] / metric[1], metric[2] / metric[3]
#     animator.add(epoch + 1, (cls_err, bbox_mae))
# print(f'class err {cls_err:.2e}, bbox mae {bbox_mae:.2e}')
# print(f'{len(train_iter.dataset) / timer.stop():.1f} examples/sec on {str(device)}')
# d2l.plt.savefig(f"{d2l.script_dir}" + "/train_loss.png")
# d2l.save_model_param(net, f"{d2l.script_dir}" + "/SSD_params.pth")

# 预测目标

d2l.load_model_param(net, f"{d2l.script_dir}" + "/SSD_params.pth")
X = torchvision.io.read_image(f"{d2l.script_dir}" + '/../img/3.jpg').unsqueeze(0).float()
# print(X.shape)
img = X.squeeze(0).permute(1, 2, 0).long()
# print(img.shape)

def predict(X):
    net.eval()
    anchors, cls_preds, bbox_preds = net(X.to(device))
    # print(anchors.shape)
    cls_probs = F.softmax(cls_preds, dim=2).permute(0, 2, 1)
    output = d2l.multibox_detection(cls_probs, bbox_preds, anchors)
    idx = [i for i, row in enumerate(output[0]) if row[0] != -1]
    return output[0, idx]

output = predict(X)
print(type(output), len(output), output.shape)

def display(img, output, threshold):
    d2l.set_figsize((8, 8))
    fig = d2l.plt.imshow(img)
    for row in output:
        score = float(row[1])
        if score < threshold:
            continue
        h, w = img.shape[0:2]
        bbox = [row[2:6] * torch.tensor((w, h, w, h), device=row.device)]
        d2l.show_bboxes(fig.axes, bbox, '%.2f' % score, 'w')

display(img, output.cpu(), threshold=0.9)
d2l.plt.savefig(f"{d2l.script_dir}" + "/banana_predict.png")