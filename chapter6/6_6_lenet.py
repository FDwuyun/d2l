import torch
from torch import nn

net = nn.Sequential(
    # in.shape = (1, 1, 28, 28)
    nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, padding=2),
    nn.ReLU(),
    # padding的作用h, w 没变 out.shape = (1, 6, 28, 28)
    nn.AvgPool2d(kernel_size=2, stride=2),
    # pooling通过空间下采样将维数减少2倍 out.shape = (1, 6, 14, 14)
    nn.Conv2d(6, 16, kernel_size=5),
    nn.ReLU(),
    # out.shape = (1, 16, 10, 10)
    nn.AvgPool2d(kernel_size=2, stride=2),
    # out.shape = (1, 16, 5, 5)
    nn.Flatten(),
    # out.shape = (1, 16 * 5 * 5 = 400)
    nn.Linear(16 * 5 * 5, 120),
    nn.ReLU(),
    # out.shape = (1, 120)
    nn.Linear(120, 84),
    nn.ReLU(),
    nn.Linear(84, 10),
)

# X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
# for layer in net:
#     X = layer(X)
#     print(layer.__class__.__name__, "output shape: \t", X.shape)
from torchinfo import summary


import torchvision
from torch.utils import data
from torchvision import transforms
import os

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
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建数据下载目录的路径（在当前脚本文件的上层目录）
    data_dir = os.path.join(script_dir, "..", "data")

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


batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size=batch_size)


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
    
    def end(self):
        """返回最后一次记录时间"""
        return sum(self.times[-235:-1])
    
    def sum(self):
        """返回时间总和"""
        return sum(self.times)

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
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = nn.CrossEntropyLoss()    
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
        test_acc = evaluate_accuracy_gpu(net, test_iter)
        print(f"{timer.end():.1f} sec,", "[epoch: {}] train_loss: {:.3f}, train_acc: {:.3f}, test_acc: {:.3f}".format(epoch, train_l, train_acc, test_acc))
        tb_writer.add_scalar("train_loss", train_l, epoch)
        tb_writer.add_scalar("train_acc", train_acc, epoch)
        tb_writer.add_scalar("test_acc", test_acc, epoch)

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
# 实例化SummaryWriter对象
script_dir = os.path.dirname(os.path.abspath(__file__))
data_dir = os.path.join(script_dir, "runs", "LeNet")
tb_writer = SummaryWriter(log_dir=data_dir)

if __name__ == '__main__':
    X = torch.rand(size=(1, 1, 28, 28), dtype=torch.float32)
    summary(net, input_data=X)
    lr, num_epochs = 0.009, 20
    tb_writer.add_graph(net, input_to_model = torch.zeros((1, 1, 28, 28)))
    train_ch6(net, train_iter, test_iter, num_epochs, lr, try_gpu())
    tb_writer.close()
