import torch
from torch import nn
from torch.nn import functional as F

class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1cov=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1cov:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(
            self.bn1(
                self.conv1(X)
            )
        )
        Y = self.bn2(
            self.conv2(Y)
        )
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    
# blk = Residual(3, 4 ,use_1x1cov=True)
# X = torch.rand(4, 3, 6, 6)
# Y = blk(X)
# print(Y.shape)

b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        # 是否是第一个残差模块的第一个残差块
        if i==0 and not first_block:
            # 不是第一个残差模块的第一个残差块，需要将上一个模块的通道数翻倍，并将高和宽减半
            blk.append(
                Residual(input_channels, num_channels, use_1x1cov=True, strides=2)
            )
        else:
            # 是第一个残差模块的第一个残差块，由于之前已经使用了步幅为2的最大汇聚层，所以无须减小高和宽
            blk.append(
                Residual(num_channels, num_channels)
            )
    return blk

b2 = nn.Sequential(
    *resnet_block(64, 64, 2, first_block=True)
)
b3 = nn.Sequential(*resnet_block(64, 128, 2))
b4 = nn.Sequential(*resnet_block(128, 256, 2))
b5 = nn.Sequential(*resnet_block(256, 512, 2))

net = nn.Sequential(
    b1, b2, b3, b4, b5,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(), 
    nn.Linear(512, 10)
)


# X = torch.rand(size=(1, 1, 224, 224))
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__,'output shape:\t', X.shape)

import torchvision
from torch.utils import data
from torchvision import transforms
import os
script_dir = os.path.dirname(os.path.abspath(__file__))

def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    # transforms.ToTensor()函数的作用是将原始的PILImage格式或者numpy.array格式的数据格式化为可被pytorch快速处理的张量类型。
    # https://blog.csdn.net/qq_38410428/article/details/94719553
    trans = [transforms.ToTensor()]  # 实例化
    if resize:
        trans.insert(0, transforms.Resize(resize))
    #  例如，我们需要对一张图片先进行尺度变换，再进行转化为Tensor算子。我们可以分步骤来，但是这样往往比较繁琐。
    # 所以，我们可以利用Compose操作。实例时，我们传入一个列表，列表分别是几个实例化后的tansforms类，作为参数传入Compose中。
    # 特别注意的是，compose中第一个操作后的数据，要符合第二个操作的输入类型。例如上中，第二个操作的输入是PIL类型，所以可以正常进行Totensor变换。
    trans = transforms.Compose(trans)
    # 获取当前脚本的绝对路径
    
    # 构建数据下载目录的路径（在当前脚本文件的上层目录）
    data_dir = os.path.join(script_dir, "..", "..", "data")

    mnist_train = torchvision.datasets.FashionMNIST(
        root=data_dir, train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root=data_dir, train=False, transform=trans, download=True
    )
    return (
        data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=2),
        data.DataLoader(mnist_test, batch_size, shuffle=False, num_workers=2),
    )

import matplotlib
matplotlib.use("Agg")  # 这一句一定要放在下面这句的前面
from matplotlib import pyplot as plt

def use_svg_display():
    """使用svg格式在Jupyter中显示绘图"""
    #可以试试加上这个代码，%config InlineBackend.figure_format = 'svg'
    # backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
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

def train_ch6(net, train_iter, test_iter, num_epochs, lr, device):
    """用GPU训练模型（在第六章定义）"""

    def init_weights(m):
        if type(m) == nn.Linear or type(m) == nn.Conv2d:
            nn.init.xavier_uniform_(m.weight)

    net.apply(init_weights)
    print("training on", device)
    net.to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()
    animator = Animator(xlabel=f'epoch, lr={lr}, batch_size={batch_size}, {device}', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])    
    timer, num_batches = Timer(), len(train_iter)
    for epoch in range(num_epochs):
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()

            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # 这部分代码计算了当前批次的损失乘以批次的大小（样本数量）。
                # 这是为了得到当前批次的总损失。通常，损失是对单个样本的损失，将其乘以批次大小可以得到批次的总损失。
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])

            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))  
        print(f"{timer.sum():.1f} sec,", "[epoch: {}] train_loss: {:.3f}, train_acc: {:.3f}, test_acc: {:.3f}".format(epoch, train_l, train_acc, test_acc))
        
        tb_writer.add_scalar("train_loss", train_l, epoch + 1)
        tb_writer.add_scalar("train_acc", train_acc, epoch + 1)
        tb_writer.add_scalar("test_acc", test_acc, epoch + 1)

    print(f"[Total] loss {train_l:.3f}, train acc {train_acc:.3f}," f"test acc {test_acc:.3f}")
    print(
        f"[Total] {timer.sum():.1f} sec, {metric[2] * num_epochs / timer.sum():.1f} examples/sec"
        f"on {str(device)}"
    )

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
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

# 获取当前日期和时间
current_datetime = datetime.datetime.now()
# 将日期和时间格式化为字符串，例如：2023-09-10-14-30-00
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
log_dir = os.path.join(script_dir, "logs", f"ResNet_{formatted_datetime}")

# 实例化SummaryWriter对象
tb_writer = SummaryWriter(log_dir = log_dir)
# 实例化Logger对象
sys.stdout = Logger(log_dir + f"/output_{formatted_datetime}.txt")

if __name__ == '__main__':

    # 输出网络结构
    X = torch.randn(size=(10, 1, 96, 96), dtype=torch.float32)
    summary(net, input_data = X)
    tb_writer.add_graph(net, input_to_model = X)
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)

    lr, num_epochs, batch_size = 0.05, 10, 256
    train_iter, test_iter = load_data_fashion_mnist(batch_size, resize=96)
    print(f'\nlearning_rate = {lr}, num_epochs = {num_epochs}, batch_size = {batch_size}\n')
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())

    tb_writer.close()

    # 获取当前日期和时间
    current_datetime = datetime.datetime.now()
    # 将日期和时间格式化为字符串，例如：2023-09-10-14-30-00
    formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
    plt.savefig(log_dir + f"/TrainChart_{formatted_datetime}.png")