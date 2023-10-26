# %%
# -*- coding: utf-8 -*-

""" 
实战Kaggle比赛：classify leaf 
https://www.kaggle.com/competitions/classify-leaves
"""
import pandas as pd
import os
from PIL import Image
import torch
from torch import nn
from torchvision import models, transforms
from torch.utils import data
from torch.utils.data import Dataset, DataLoader

# %%
# 一、下载数据集 
# kaggle competitions download -c classify-leaves

# 二、读取数据集
# current_file_dir = os.getcwd()
script_dir = os.path.dirname(os.path.abspath(__file__))
# print(current_file_dir)

# %%
"""
训练数据集包括18353个样本，每个样本个特征和1个标签，
而测试数据集包含8800个样本，
共176种类
"""
train_data = pd.read_csv(os.path.join(script_dir, "data", "train.csv"))
# print(train.shape)
# print(test.shape)

# %%

train_labels = list(train_data["label"])
# 获取训练集中标签的种类
labels_categories = list(set(train_labels))
# 将训练集对应的标签转化为数字索引 labels_categories -> index
labels_num = []
for i in range(len(train_labels)):
    labels_num.append(labels_categories.index(train_labels[i]))

train_data["number"] = labels_num
# print(train.shape)
# index=False 不在文件的第一列加索引
# train.to_csv("./data/train_num_label.csv", index=False) 

# %%
# 三、预处理数据
class Leaf_Train_Dataset(Dataset):
    '''
    树叶数据集的训练集 自定义Dataset
    '''
    def __init__(self, train_X, train_y, train_path = None, transform = None) -> None:
        '''
        train_path : 传入记录图像路径及其标号的csv文件
        transform : 对图像进行的变换
        '''
        super().__init__()
        if train_path != None:
            train_data = pd.read_csv(train_path)
            train_X = train_data["image"]
            train_y = train_data["number"]

        # 以列表的形式记录图像所在地址
        self.images_list = train_X
        # 图像的标号记录
        self.label_nums = train_y
        self.transform = transform

    def __getitem__(self, idx):
        '''
        idx : 所需要获取的图像的索引
        return : image， label
        '''
        image = Image.open(os.path.join("/home/qlf/d2l/chapter7/7_8_classify_leaf/data", self.images_list[idx]))
        if(self.transform != None):
            image = self.transform(image)
        label = self.label_nums[idx]
        return image, label
    
    def __len__(self):
        return len(self.images_list)

class Leaf_Test_Dataset(Dataset):

    def __init__(self, test_X_list, transform = None):

        super().__init__()

        # 以列表的形式记录图像所在地址
        self.images_list = test_X_list

        self.transform = transform

    def __getitem__(self, idx):

        image = Image.open(os.path.join("/home/qlf/d2l/chapter7/7_8_classify_leaf/data", self.images_list[idx]))
        if(self.transform != None):
            image = self.transform(image)
        return image
    
    def __len__(self):
        return len(self.images_list)

transforms_train = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        #随机水平翻转 选择一个概率
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
        transforms.ToTensor()
    ]
)
transforms_test = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ]
)


# %%
import matplotlib
matplotlib.use("Agg")  # 这一句一定要放在下面这句的前面
from matplotlib import pyplot as plt

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

# %%
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

log_dir = os.path.join(script_dir, "logs", f"Net_{get_datetime()}")

# 实例化SummaryWriter对象
tb_writer = SummaryWriter(log_dir = log_dir)
# 实例化Logger对象
sys.stdout = Logger(log_dir + f"/output_{get_datetime()}.txt")


# %%
# 训练
def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")

def train(net, train_loader, valid_loader, num_epochs, lr, weight_decay, device):
    timer2 = Timer()
    timer2.start()

    if(net.parameters().__next__().device != device):
        net.to(device)
        print(f"[{get_datetime()}] training on ", device)
    optimizer = torch.optim.SGD(net.parameters(),
                                 lr=lr, weight_decay=weight_decay)
    loss = nn.CrossEntropyLoss()

    animator = Animator(xlabel=f'epoch, lr={lr}, {device}', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'valid acc'])    
    
    # num_batches 表示 total_samples / batch_size
    num_batches = len(train_loader)
    print(f"[{get_datetime()}] Each epoch includes {num_batches} batches")
    timer  = Timer()
    for epoch in range(num_epochs):
        timer.start()
        net.train()
        # 训练损失之和，训练准确率之和，样本数
        metric = Accumulator(3)
        # 循环的次数为 num_batches
        
        for i, (X, y) in enumerate(train_loader):
            
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.backward()
            optimizer.step()
            with torch.no_grad():
                # 这部分代码计算了当前批次的损失乘以批次的大小（样本数量）。
                # 这是为了得到当前批次的总损失。通常，损失是对单个样本的损失，将其乘以批次大小可以得到批次的总损失。
                # X.shape[0] 经常= batch_size ，但最后一个loader一般小于batch_size
                metric.add(l * X.shape[0], accuracy(y_hat, y), X.shape[0])
            
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                train_loss = metric[0] / metric[2]
                train_acc = metric[1] / metric[2]
                animator.add(
                    epoch + (i + 1) / num_batches,
                    (train_loss, train_acc, None)
                )
        valid_acc = evaluate_accuracy_gpu(net, valid_loader)

        animator.add(epoch + 1, (None, None, valid_acc))
        tb_writer.add_scalar("train_loss", train_loss, epoch + 1)
        tb_writer.add_scalar("train_acc", train_acc, epoch + 1)
        tb_writer.add_scalar("valid_acc", valid_acc, epoch + 1)

        timer.stop()  
        print(f"[{get_datetime()}] {timer.sum():.1f} sec,", "[epoch: {}] train_loss: {:.3f}, train_acc: {:.3f}, valid_acc: {:.3f}".format(epoch + 1, train_loss, train_acc, valid_acc))
        print(f"[Total] loss {train_loss:.3f}, train acc {train_acc:.3f}," f"valid acc {valid_acc:.3f}")
        # 保存模型参数
        # net_param_sava_path = log_dir + f"/resnet_{get_datetime()}.pth"
        # save_model_param(net, net_param_sava_path)
    timer2.stop()
    print(
        f"[Total {num_epochs} epochs] {timer2.sum():.1f} sec, {metric[2] * num_epochs / timer.sum():.1f} examples/sec"
        f"on {str(device)}"
    )
    return train_loss, train_acc, valid_acc 

"""
[**K折交叉验证**]有助于模型选择和超参数调整。
在$K$折交叉验证过程中返回第$i$折的数据。
具体地说，它选择第$i$个切片作为验证数据，其余部分作为训练数据。
注意，这并不是处理数据的最有效方法，如果我们的数据集大得多，会有其他解决办法。
"""

def get_k_fold_dataset(k, i, X, y):
    assert k > 1 , "K折交叉验证 require k > 1"
    # //是向下取整
    fold_size = len(X) // k 
    X_train, y_train = None, None
    for j in range(k):
        # slice() 函数实现切片对象，主要用在切片操作函数里的参数传递。
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx], y[idx]
        
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            # torch.cat(tensors, dim=0, out=None) -> Tensor, 一个将张量沿着指定维度拼接起来的函数
            X_train.extend(X_part)
            y_train.extend(y_part)
    
    train_dataset = Leaf_Train_Dataset(X_train, y_train, None, transforms_train)
    valid_dataset = Leaf_Train_Dataset(X_valid, y_valid, None, transforms_train)
    
    return train_dataset, valid_dataset

def k_fold_cross_train(net, train_data, k, num_epochs, lr, weight_decay, batch_size, device):
    train_loss, train_acc, valid_acc = [], [], []
    # train_loss_sum, train_acc_sum, valid_acc_sum = 0, 0, 0
    timer1  = Timer()
    for i in range(k):
        timer1.start()
        X, y = train_data["image"], train_data["number"]

        train_dataset, valid_dataset = get_k_fold_dataset(k, i, list(X), list(y))
        train_loader = data.DataLoader(train_dataset, batch_size, shuffle=True)
        valid_loader = data.DataLoader(valid_dataset, batch_size, shuffle=True)
        
        print(f'[{get_datetime()}] [ {i + 1} 折]')

        a, b, c = train(net, train_loader, valid_loader, num_epochs, lr, weight_decay, device)
        train_loss.append(a), train_acc.append(b), valid_acc.append(c)
        timer1.stop()
        print(f"[{get_datetime()}] Total [ {i + 1} 折], {timer1.sum():.4f} sec ")

    print(f"[{get_datetime()}] [K折交叉训练平均] train_loss {sum(train_loss) / k:.3f}, train acc {sum(train_acc) / k:.3f}," f"valid acc {sum(valid_acc) / k:.3f}")
        
    return sum(train_loss) / k, sum(train_acc) / k, sum(valid_acc) / k


# %%
# 选择网络

resnet50 = models.resnet50(weights = models.ResNet50_Weights.IMAGENET1K_V1)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_uniform_(m.weight)

net = resnet50

# %%开始训练
if __name__ == '__main__':
    # 输出网络结构
    X = torch.randn(size=(10, 3, 224, 224), dtype=torch.float32)
    summary(net, input_data = X)
    k, num_epochs, lr, weight_decay, batch_size = 5, 50, 0.01, 1e-3, 128
    print(f"\n k, num_epochs, lr, weight_decay, batch_size = {k, num_epochs, lr, weight_decay, batch_size} \n")

    load_model_param(net, "/home/qlf/d2l/chapter7/7_8_classify_leaf/logs/Net_2023-10-07-22-00-12/resnet_2023-10-08-01-57-11.pth")
    k_fold_cross_train(net, train_data, k, num_epochs, lr, weight_decay, batch_size, try_gpu())
    
    tb_writer.close()
    plt.savefig(log_dir + f"/TrainChart_{get_datetime()}.png")



# %%
# 预测并生成csv文件

test_data = pd.read_csv(os.path.join(script_dir, "data", "test.csv"))

test_dateset = Leaf_Test_Dataset(list(test_data["image"]), transforms_test)
test_loader = DataLoader(test_dateset, batch_size=64, shuffle=False)

def predict(net, test_loader, device):
    print("testing on ", device)
    # 设置模型为评估模式,
    net.eval() 
    net.to(device)
    
    predict_y = torch.tensor([]).to(device)

    with torch.no_grad():
        for X in test_loader:
            X = X.to(device)
            y_hat = net(X)
            predict_y = torch.cat((predict_y, y_hat), dim=0)
        # reshape(-1)会将多维张量变为一维张量
        predict_y = torch.argmax(predict_y, dim=1).reshape(-1)
    predict_label = []
    # print(predict_y.shape[0], len(labels_categories))
    for i in range(predict_y.shape[0]):
        idx = predict_y[i]
        if(idx < 0):
            predict_label.append(labels_categories[0])
        elif(idx >= 176):
            predict_label.append(labels_categories[175])
        else:
            predict_label.append(labels_categories[idx])

    
    # 将其重新格式化以导出到Kaggle
    test_data["label"] = pd.Series(predict_label)
    test_data.to_csv(os.path.join(script_dir, 'submission.csv'), index=False)
    print("已生成submission.csv in ")

predict(net, test_loader, try_gpu())
