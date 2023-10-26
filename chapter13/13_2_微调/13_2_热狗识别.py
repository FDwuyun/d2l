import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models,transforms,datasets
from torch.utils import data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = os.getcwd()

import hashlib
import tarfile
import zipfile
import requests


DATA_URL = "http://d2l-data.s3-accelerate.amazonaws.com/"
DATA_HUB = dict()
DATA_HUB["hotdog"] = (
    DATA_URL + "hotdog.zip", "fba480ffa8aa7e0febbb511d181409f899b9baa5",
)

def download(name, cache_dir=os.path.join(script_dir, "..", "..", "data")):
    """下载一个DATA_HUB中的文件，返回本地文件名"""
    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"
    url, sha1_hash = DATA_HUB[name]
    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split("/")[-1])
    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, "rb") as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f"正在从{url}下载{fname}...")
    r = requests.get(url, stream=True, verify=True)
    with open(fname, "wb") as f:
        f.write(r.content)
    print(f"{fname}下载完毕")
    return fname


def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == ".zip":
        fp = zipfile.ZipFile(fname, "r")
    elif ext in (".tar", ".gz"):
        fp = tarfile.open(fname, "r")
    else:
        assert False, "只有zip/tar文件可以被解压缩"
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

# data_dir = download_extract("hotdog")


import matplotlib
matplotlib.use("Agg")  # 这一句一定要放在下面这句的前面
from matplotlib import pyplot as plt

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            # 图片张量
            ax.imshow(img.numpy())
        else:
            # PIL图片
            ax.imshow(img)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

# show_images(
#     [train_imgs[i][0] for i in range(8)] + [train_imgs[-i-1][0] for i in range(8)],
#     2,
#     8
# )
# plt.savefig(script_dir + f"/Doghot.png")

import numpy as np
def compute_mean_std(dataset):
    """
    计算数据集的RGB通道均值和标准差
    Args:
        dataset: 数据集
    Returns:
        均值和标准差的列表
    """
    mean = np.zeros(3)
    std = np.zeros(3)
    n_images = 0

    for img, _ in dataset:
        img = np.array(img)
        mean += np.mean(img, axis=(0, 1))
        std += np.std(img, axis=(0, 1))
        n_images += 1

    mean /= n_images
    std /= n_images

    return mean.tolist(), std.tolist()

# 计算均值和标准差
# mean, std = compute_mean_std(train_imgs)
# print("Mean:", mean)
# print("Std:", std)

# 使用RGB通道的均值和标准差，以标准化每个通道
normalize = transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
)


def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    #可以试试加上这个代码，%config InlineBackend.figure_format = 'svg'
    # backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(7, 5)):
    """设置matplotlib的图表大小"""
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
    """设置matplotlib的轴"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    if legend:
        axes.legend(legend)
    axes.grid()

#通过以上三个用于图形配置的函数，定义一个plot函数来简洁地绘制多条曲线， 因为我们需要在整个书中可视化许多曲线。
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
        legend = []

    set_figsize(figsize)
    axes = axes if axes else plt.gca()

    # 如果X有一个轴，输出True
    def has_one_axis(X):
        return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
                and not hasattr(X[0], "__len__"))

    if has_one_axis(X):
        X = [X]
    if Y is None:
        X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
        Y = [Y]
    if len(X) != len(Y):
        X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
        if len(x):
            axes.plot(x, y, fmt)
        else:
            axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)


class Animator:
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
                 ylim=None, xscale='linear', yscale='linear',
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
                 figsize=(7, 5)):
        # 增量地绘制多条线
        if legend is None:
            legend = []
        use_svg_display()
        self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: set_axes(
            self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla()
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        plt.show()

class Accumulator:
    """在n个变量上累加"""

    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)  # 获得每行中最大元素的索引来获得预测类别
    cmp = y_hat.type(y.dtype) == y  #
    return float(cmp.type(y.dtype).sum())  # 返回预测正确的个数

import time
import numpy as np

class Timer:
    """记录多次运行时间"""

    def __init__(self):
        self.times = []
        self.lastTimeSum = 0
        self.start()

    def start(self):
        """启动计时器"""
        self.tik = time.time()

    def stop(self):
        """停止计时器并将时间记录在列表中"""
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        """返回平均时间"""
        return sum(self.times) / len(self.times)
    
    def sum(self):
        """返回时间总和"""
        self.lastTimeSum = sum(self.times)
        return self.lastTimeSum

    def cumsum(self):
        """返回累计时间"""
        return np.array(self.times).cumsum().tolist()

def evaluate_accuracy_gpu(net, data_iter, device=None):
    """使用GPU计算模型在数据集上的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()  # 设置为评估模式，关闭Dropout和直接结算所有batch的均值和方差
        if not device:
            # 使用参数来构建一个虚拟的计算图，然后从计算图中获取一个参数张量，然后通过 .device 属性获取这个参数张量所在的设备。这个参数张量位于模型的第一个参数（通常是一个权重矩阵）。
            device = next(iter(net.parameters())).device
    # 正确预测的数量，总预测的数量
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                # BERT微调所需要的
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

# 保存模型和参数
def save_model_param(model, save_path):
    torch.save(model.state_dict(), save_path)
    print(f"模型参数保存完毕 in {save_path}")

def load_model_param(model, save_path):
    model.load_state_dict(torch.load(save_path))
    print(f"模型参数加载完毕 in {save_path}")
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if i + 1 <= torch.cuda.device_count():
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

from tensorboardX import SummaryWriter
from torchinfo import summary
import datetime
import sys

class Logger():

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

def get_datetime():
    # 获取当前日期和时间
    current_datetime = datetime.datetime.now()
    # 将日期和时间格式化为字符串，例如：2023-09-10-14-30-00
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    return formatted_datetime

transforms_train = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ]
)
transforms_test = transforms.Compose(
    [
        transforms.Resize([256, 256]),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ]
)
HOTDOG_DATASET_PATH = script_dir + "/../../data/hotdog"
train_datasets = datasets.ImageFolder(os.path.join(HOTDOG_DATASET_PATH, "train"), transform=transforms_train)
test_datasets = datasets.ImageFolder(os.path.join(HOTDOG_DATASET_PATH, "test"), transform=transforms_test)

# pretrained_net = torchvision.models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)
# print(pretrained_net.fc)

def train_batch_ch13(net, X, y, loss, optimizer, devices):
    # 用多GPU进行小批量训练
    if isinstance(X, list):
        # 微调BERT中所需
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    
    y = y.to(devices[0])

    net.train()

    optimizer.zero_grad()
    y_hat = net(X)
    l = loss(y_hat, y)
    l.backward()
    optimizer.step()

    train_loss_sum = l
    train_acc_sum = accuracy(y_hat, y)
    return train_loss_sum, train_acc_sum

def train_ch13(net, train_loader, valid_loader, loss, optimizer, num_epochs, num_gpus):
    timer2 = Timer()
    timer2.start()

    devices = [try_gpu(i) for i in range(num_gpus)]
    print(f"[{get_datetime()}] training on ", devices)

    # 用多GPU进行模型训练
    timer = Timer()
    num_batches = len(train_loader)
    animator = Animator(xlabel=f'epoch, {devices}', xlim=[1, num_epochs], ylim=[0, 1],
                        legend=["train loss", "train acc", "test acc"])
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    print(f"[{get_datetime()}] Each epoch includes {num_batches} batches")
    for epoch in range(num_epochs):
        timer.start()
        net.train()
        # 4个维度：储存训练损失，训练准确度，实例数，特点数
        metric = Accumulator(4)
        for i, (X, y) in enumerate(train_loader):
            l, acc = train_batch_ch13(net, X, y, loss, optimizer, devices)
            metric.add(l, acc, X.shape[0], X.numel())
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                animator.add(
                    epoch + ( i + 1) / num_batches,
                    (train_loss, train_acc, None)
                )
        valid_acc = evaluate_accuracy_gpu(net, valid_loader)
        animator.add(epoch + 1, (None, None, valid_acc))
        timer.stop()
        print(f"[{get_datetime()}][epoch: {epoch+1}] {timer.sum():.1f} sec, train_loss {train_loss:.3f}, train_acc {train_acc:.3f}, valid_acc {valid_acc:.3f}")
    timer2.stop()
    print(f"[Total {num_epochs} epochs] {timer2.sum():.1f} sec, train_loss {metric[0] / metric[2]:.3f}, train_acc {metric[1] / metric[2]:.3f}, valid_acc{valid_acc:.3f}")
    print(f"{metric[2] * num_epochs / timer.sum():.1f} examples/sec on {str(devices)}")


# 如果param_group=True，输出层中的模型参数将使用十倍的学习率
def train_fine_tuning(net, num_epochs, batch_size, num_gpus, lr, weight_decay, param_group=True):
    train_iter = data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    test_iter = data.DataLoader(test_datasets, batch_size=batch_size)

    loss = nn.CrossEntropyLoss(reduction="sum")
    if param_group:
        #  params_1x 包含了除了 "fc.weight" 和 "fc.bias" 之外的所有网络参数
        params_1x = [param for name, param in net.named_parameters()
                     if name not in ["fc.weight", "fc.bias"]]
        trainer = torch.optim.SGD(
            [
                {'params': params_1x},
                {'params': net.fc.parameters(), 'lr': lr*10}
            ],
            lr=lr,
            weight_decay=weight_decay
        )
    else:
        trainer = torch.optim.SGD(
            net.parameters(),
            lr = lr,
            weight_decay=weight_decay
        )
    train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, num_gpus)


finetune_net = models.resnet18(weights = models.ResNet18_Weights.IMAGENET1K_V1)

hotdog_w = torch.split(finetune_net.fc.weight.data, 1, dim=0)[934]
finetune_net.fc = nn.Linear(finetune_net.fc.in_features, 2)
with torch.no_grad():
    finetune_net.fc.weight[0] = hotdog_w
nn.init.xavier_uniform_(finetune_net.fc.weight)

ps = {
    "net": "models_resnet18",
    "weights": "weights = models.ResNet18_Weights.IMAGENET1K_V1",
    "num_gpus": 3,
    "num_epochs": 10,
    "batch_size": 128,
    "lr": 5e-5,
    "weight_decay": 0.001,
    "description": "微调，热狗识别,finetune_net.fc.weight[0] = hotdog_w"
}

log_dir = os.path.join(script_dir, "logs", f"{ps['net']}_{get_datetime()}")
# 实例化SummaryWriter对象
tb_writer = SummaryWriter(log_dir = log_dir)
# 实例化Logger对象
sys.stdout = Logger(log_dir + f"/output_{get_datetime()}.txt")

if __name__ == '__main__':
    print(ps)
    train_fine_tuning(
        finetune_net, ps["num_epochs"], ps["batch_size"],
          ps["num_gpus"], ps["lr"], ps["weight_decay"], param_group=True
    )
    plt.savefig(script_dir + f"/{ps['net']}.png")