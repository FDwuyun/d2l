import hashlib
import os
import tarfile
import zipfile
import requests
import numpy as np
import pandas as pd
import torch
from torch import nn

script_dir = os.path.dirname(os.path.abspath(__file__))

DATA_HUB = dict()
DATA_URL = 'http://d2l-data.s3-accelerate.amazonaws.com/'

def download(name, cache_dir=os.path.join(script_dir, "..", "..", 'data')):
    """下载一个DATA_HUB中的文件，返回本地文件名"""

    assert name in DATA_HUB, f"{name} 不存在于 {DATA_HUB}"

    url, sha1_hash = DATA_HUB[name]

    os.makedirs(cache_dir, exist_ok=True)
    fname = os.path.join(cache_dir, url.split('/')[-1])

    if os.path.exists(fname):
        sha1 = hashlib.sha1()
        with open(fname, 'rb') as f:
            while True:
                data = f.read(1048576)
                if not data:
                    break
                sha1.update(data)
        if sha1.hexdigest() == sha1_hash:
            return fname  # 命中缓存
    print(f'正在从{url}下载{fname}...')
    r = requests.get(url, stream=True, verify=True)
    with open(fname, 'wb') as f:
        f.write(r.content)
    print(f'{fname}下载完毕')
    return fname

"""我们还需实现两个实用函数：
一个将下载并解压缩一个zip或tar文件，
另一个是将本书中使用的所有数据集从`DATA_HUB`下载到缓存目录中。
"""
def download_extract(name, folder=None):
    """下载并解压zip/tar文件"""
    fname = download(name)
    base_dir = os.path.dirname(fname)
    data_dir, ext = os.path.splitext(fname)
    if ext == '.zip':
        fp = zipfile.ZipFile(fname, 'r')
    elif ext in ('.tar', '.gz'):
        fp = tarfile.open(fname, 'r')
    else:
        assert False, '只有zip/tar文件可以被解压缩'
    fp.extractall(base_dir)
    return os.path.join(base_dir, folder) if folder else data_dir

def download_all():
    """下载DATA_HUB中的所有文件"""
    for name in DATA_HUB:
        download(name)



"""为方便起见，我们可以使用上面定义的脚本下载并缓存Kaggle房屋数据集。

"""

DATA_HUB['kaggle_house_train'] = (
        DATA_URL + 'kaggle_house_pred_train.csv',
        '585e9cc93e70b39160e7921475f9bcd7d31219ce'
    )

DATA_HUB['kaggle_house_test'] = (
        DATA_URL + 'kaggle_house_pred_test.csv',
        'fa19780a7b011d9b009e8bff8e99922a8ee2eb90'
    )

"""
我们使用`pandas`分别加载包含训练数据和测试数据的两个CSV文件。
"""
train_data = pd.read_csv(download('kaggle_house_train'))
test_data = pd.read_csv(download('kaggle_house_test'))

"""
训练数据集包括1460个样本，每个样本80个特征和1个标签，
而测试数据集包含1459个样本，每个样本80个特征。
"""
# print(train_data.shape)
# print(test_data.shape)


"""
我们可以看到，(**在每个样本中，第一个特征是ID，**)
这有助于模型识别每个训练样本。
虽然这很方便，但它不携带任何用于预测的信息。
因此，在将数据提供给模型之前，(**我们将其从数据集中删除**)。
"""
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
# print(all_features.shape)
# print(all_features.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])
all_features.dtypes, all_features.dtypes != 'object', all_features.dtypes[all_features.dtypes != 'object'].index

# 若无法获得测试数据，则可根据训练数据计算均值和标准差
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
# 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[numeric_features] = all_features[numeric_features].fillna(0)
# print(all_features['YrSold'].mean(), all_features['MSSubClass'].std())

"""
接下来，我们[**处理离散值。**]
这包括诸如“MSZoning”之类的特征。
(**我们用独热编码替换它们**)，
方法与前面将多类别标签转换为向量的方式相同
（请参见 :numref:`subsec_classification-problem`）。
例如，“MSZoning”包含值“RL”和“Rm”。
我们将创建两个新的指示器特征“MSZoning_RL”和“MSZoning_RM”，其值为0或1。
根据独热编码，如果“MSZoning”的原始值为“RL”，
则：“MSZoning_RL”为1，“MSZoning_RM”为0。
`pandas`软件包会自动为我们实现这一点。
"""

# “Dummy_na=True”将“na”（缺失值）视为有效的特征值，并为其创建指示符特征
all_features = pd.get_dummies(all_features, dummy_na=True)
# print(all_features.shape)
# print(type(train_data), type(all_features), train_data.shape, train_data.shape[0], train_data['SalePrice'])

n_train = train_data.shape[0]
train_features = torch.tensor(all_features[:n_train].values.astype(np.float32), dtype=torch.float32)
test_features = torch.tensor(all_features[n_train:].values.astype(np.float32), dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32
)

"""
模型选择
"""
# print(train_features.shape, train_features.shape[1])
in_features = train_features.shape[1]

def get_LinearNet():
    net = nn.Sequential(nn.Linear(in_features,1))
    return net

def get_MLPNet():
    net = nn.Sequential(
          nn.Flatten(),
          nn.Linear(in_features, 1024),
          nn.ReLU(),
          nn.Dropout(0.2),
          nn.Linear(1024, 1))
    return net

def linear_layer(input_features, num_features):
    return nn.Sequential(
        nn.Linear(input_features, num_features),
        nn.BatchNorm1d(num_features), nn.ReLU(),
        nn.Dropout(0.5)
    )

class DenseBlock(nn.Module):
    def __init__(self, num_layers, input_features, num_features):
        super().__init__()
        layer = []
        for i in range(num_layers):
            layer.append(
                linear_layer(num_features * i + input_features, num_features)
            )
        self.net = nn.Sequential(*layer)

    def forward(self, X):
        for layer in self.net:
            Y = layer(X)
            # 连接特征维度上的每层的输入和输出
            X = torch.cat(
                (X, Y),
                dim=1
            )
        return X

def transition_layer(input_features, num_features):
    return nn.Sequential(
        
        nn.Linear(input_features, num_features), nn.BatchNorm1d(num_features), nn.ReLU()
    )

b1 = nn.Sequential(
    nn.Linear(in_features, 512), nn.ReLU(),
    nn.Linear(512, 1024), nn.ReLU()
)

# num_features为当前的特征数
num_features, growth_rate = 1024, 128
num_layers_in_dense_blocks = [2, 4, 4, 2]
blks = []
for i, num_layers in enumerate(num_layers_in_dense_blocks):
    blks.append(DenseBlock(num_layers, num_features, growth_rate))
    # 上一个稠密块的输出特征数
    num_features += num_layers * growth_rate
    # 在稠密块之间添加一个过渡层，使通道数量减半
    if i != len(num_layers_in_dense_blocks) - 1:
        blks.append(transition_layer(num_features, num_features // 2))
        num_features = num_features // 2

def get_DenseNet():
    net = nn.Sequential(
        b1,
        *blks,
        nn.BatchNorm1d(num_features), nn.ReLU(),
        nn.Linear(num_features, 1)
    )
    return net 

net = get_DenseNet()

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
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(7, 5), axes=None):
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

"""房价就像股票价格一样，我们关心的是相对数量，而不是绝对数量。
因此，[**我们更关心相对误差$\frac{y - \hat{y}}{y}$，**]
而不是绝对误差$y - \hat{y}$。
例如，如果我们在俄亥俄州农村地区估计一栋房子的价格时，
假设我们的预测偏差了10万美元，
然而那里一栋典型的房子的价值是12.5万美元，
那么模型可能做得很糟糕。
另一方面，如果我们在加州豪宅区的预测出现同样的10万美元的偏差，
（在那里，房价中位数超过400万美元）
这可能是一个不错的预测。

(**解决这个问题的一种方法是用价格预测的对数来衡量差异**)。
事实上，这也是比赛中官方用来评价提交质量的误差指标。
即将$\delta$ for $|\log y - \log \hat{y}| \leq \delta$
转换为$e^{-\delta} \leq \frac{\hat{y}}{y} \leq e^\delta$。
这使得预测价格的对数与真实标签价格的对数之间出现以下均方根误差：

$$\sqrt{\frac{1}{n}\sum_{i=1}^n\left(\log y_i -\log \hat{y}_i\right)^2}.$$

"""

loss = nn.MSELoss()
def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    # torch.clamp(input, min, max, out=None)将输入input张量每个元素的范围限制到区间[min,max]，小于min的元素会被min替代，大于max的元素被max替代，返回结果到一个新张量。
    features, labels = features.to(try_gpu()), labels.to(try_gpu())
    clipped_preds = torch.clamp(
        input = net(features),
        min = 1,
        max = float('inf')
    )
    rmse = torch.sqrt(
        loss(
            torch.log(clipped_preds), torch.log(labels)
        )
    )
    return rmse.item()

"""与前面的部分不同，[**我们的训练函数将借助Adam优化器**]
（我们将在后面章节更详细地描述它）。
Adam优化器的主要吸引力在于它对初始学习率不那么敏感。
"""

from torch.utils import data

def load_array(data_arrays, batch_size, is_train=True):
    """构造一个PyTorch数据迭代器"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def train(net, train_features, train_labels, valid_features, valid_labels,
          num_epochs, learning_rate, weight_decay, batch_size, device):

    train_loss, valid_loss = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    
    timer = Timer()
    timer.start()

    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            l = loss(net(X), y) # l是标量
            l.backward()
            optimizer.step()
        train_loss.append(log_rmse(net, train_features, train_labels))

        if valid_labels is not None:
            valid_loss.append(log_rmse(net, valid_features, valid_labels))
    timer.stop()
    print(f"{timer.sum():.1f} sec")
    return train_loss, valid_loss

"""## $K$折交叉验证

本书在讨论模型选择的部分（ :numref:`sec_model_selection`）
中介绍了[**K折交叉验证**]，
它有助于模型选择和超参数调整。
我们首先需要定义一个函数，在$K$折交叉验证过程中返回第$i$折的数据。
具体地说，它选择第$i$个切片作为验证数据，其余部分作为训练数据。
注意，这并不是处理数据的最有效方法，如果我们的数据集大得多，会有其他解决办法。

"""

def get_k_fold_data(k, i, X, y):
    assert k > 1 , "require k > 1"
    fold_size = X.shape[0] // k # //是向下取整
    X_train, y_train = None, None
    for j in range(k):
        # slice() 函数实现切片对象，主要用在切片操作函数里的参数传递。
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            # torch.cat(tensors, dim=0, out=None) -> Tensor, 一个将张量沿着指定维度拼接起来的函数
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def k_fold_cross_train(k, X_train, y_train, num_epochs, learning_rate, weight_decay, batch_size):
        
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    
    train_loss_sum, valid_loss_sum = 0, 0
   
    print("training on", try_gpu())
    net.to(try_gpu())
    net.apply(init_weights)

    for i in range(k):
        data = get_k_fold_data(k, i, X_train, y_train)
        
        train_loss, valid_loss = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size, try_gpu())
        train_loss_sum += train_loss[-1] # 只要每折最后一轮的损失
        valid_loss_sum += valid_loss[-1]
        if i == 0:
            plot(X = list(range(1, num_epochs + 1)), Y = [train_loss, valid_loss],
                     xlabel='epoch', ylabel='rmse',
                     xlim=[1, num_epochs],
                     legend=['train', 'valid'])
        
        print(f'折{i + 1}，训练log rmse{float(train_loss[-1]):f}, 验证log rmse{float(valid_loss[-1]):f}')
    return train_loss_sum / k, valid_loss_sum / k

"""## [**模型选择**]

在本例中，我们选择了一组未调优的超参数，并将其留给读者来改进模型。
找到一组调优的超参数可能需要时间，这取决于一个人优化了多少变量。
有了足够大的数据集和合理设置的超参数，$K$折交叉验证往往对多次测试具有相当的稳定性。
然而，如果我们尝试了不合理的超参数，我们可能会发现验证效果不再代表真正的误差。

"""
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
log_dir = os.path.join(script_dir, "logs", f"kaggle_{formatted_datetime}")
# 实例化SummaryWriter对象
tb_writer = SummaryWriter(log_dir = log_dir)
# 实例化Logger对象
sys.stdout = Logger(log_dir + f"/output_{formatted_datetime}.txt")


# __main__
# 输出网络结构
X = torch.randn(size=(10, in_features), dtype=torch.float32)
summary(net, input_data = X)

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.1, 0, 128
print(f'\nk, num_epochs, lr, weight_decay, batch_size = {k, num_epochs, lr, weight_decay, batch_size} \n')
    
train_l, valid_l = k_fold_cross_train(k, train_features, train_labels, num_epochs, lr,
                          weight_decay, batch_size)
print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
      f'平均验证log rmse: {float(valid_l):f}')

tb_writer.close()
# 获取当前日期和时间
current_datetime = datetime.datetime.now()
# 将日期和时间格式化为字符串，例如：2023-09-10-14-30-00
formatted_datetime = current_datetime.strftime("%Y-%m-%d-%H-%M-%S")
plt.savefig(log_dir + f"/TrainChart_{formatted_datetime}.png")

"""请注意，有时一组超参数的训练误差可能非常低，但$K$折交叉验证的误差要高得多，
这表明模型过拟合了。
在整个训练过程中，我们希望监控训练误差和验证误差这两个数字。
较少的过拟合可能表明现有数据可以支撑一个更强大的模型，
较大的过拟合可能意味着我们可以通过正则化技术来获益。

##  [**提交Kaggle预测**]

既然我们知道应该选择什么样的超参数，
我们不妨使用所有数据对其进行训练
（而不是仅使用交叉验证中使用的$1-1/K$的数据）。
然后，我们通过这种方式获得的模型可以应用于测试集。
将预测保存在CSV文件中可以简化将结果上传到Kaggle的过程。

"""

def train_and_pred(net, train_features, train_labels, test_features, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    

    train_loss, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size, try_gpu())
    plot(X=np.arange(1, num_epochs + 1), Y=[train_loss], xlabel='epoch',
             ylabel='log rmse', xlim=[1, num_epochs])
    print(f'训练log rmse：{float(train_loss[-1]):f}')
    
    # 将网络应用于测试集。
    test_features = test_features.to(try_gpu())
    preds = net(test_features).cpu().detach().numpy()
    # 将其重新格式化以导出到Kaggle
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv(os.path.join(script_dir, 'submission.csv'), index=False)

"""
如果测试集上的预测与$K$倍交叉验证过程中的预测相似，
那就是时候把它们上传到Kaggle了。
下面的代码将生成一个名为`submission.csv`的文件。

"""

train_and_pred(net, train_features, train_labels, test_features, test_data, num_epochs, lr, weight_decay, batch_size)
