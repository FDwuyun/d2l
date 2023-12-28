import torch
from torch import nn
import d2l_13 as d2l

def trans_conv(X, K):
    h, w = K.shape
    Y = torch.zeros((X.shape[0] + h - 1, X.shape[1] + w - 1))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Y[i: i + h, j: j + w] += X[i, j] * K
    return Y

X = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
# print(trans_conv(X, K))

X, K = X.reshape(1, 1, 2, 2), K.reshape(1, 1, 2, 2)
tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, bias=False)
tconv.weight.data = K
# print(tconv(X))

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, padding=1, bias=False)
tconv.weight.data = K
# print(tconv(X))

tconv = nn.ConvTranspose2d(1, 1, kernel_size=2, stride=4, padding=0, bias=False) # 2*4+2-0-4 = 6
tconv.weight.data = K
# print(tconv(X))

X = torch.rand(size=(1, 10, 16, 16))
conv = nn.Conv2d(10, 20, kernel_size=5, padding=2, stride=3)
# (16-5+4+3)/3 = (16-5+4)/3+1 = 6, torch.Size([1, 20, 6, 6])
# print(conv(X).shape)
tconv = nn.ConvTranspose2d(20, 10, kernel_size=5, padding=2, stride=3)
# (3*6+5-4-3) = (6-1)*3+5-4 = 16
# print(tconv(conv(X)).shape == X.shape)

X = torch.arange(9.0).reshape(3, 3)
K = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
Y = d2l.corr2d(X, K) # 卷积
print(Y)

def kernel2matrix(K):
    k, W = torch.zeros(5), torch.zeros((4, 9))
    k[:2], k[3:5] = K[0, :], K[1, :]
    W[0, :5], W[1, 1:6], W[2, 3:8], W[3, 4:] = k, k, k, k
    return W
W = kernel2matrix(K)
print(W)
# 逐行连结输入X，获得了一个长度为9的矢量。 然后，W的矩阵乘法和向量化的X给出了一个长度为4的向量。 重塑它之后，可以获得与上面的原始卷积操作所得相同的结果Y：我们刚刚使用矩阵乘法实现了卷积。
print(Y == torch.matmul(W, X.reshape(-1)).reshape(2, 2))

# 同样，我们可以使用矩阵乘法来实现转置卷积。 在下面的示例中，我们将上面的常规卷积
# 的输出Y作为转置卷积的输入。 想要通过矩阵相乘来实现它，我们只需要将权重矩阵W的形状转置为。
Z = trans_conv(Y, K)
print(Z == torch.matmul(W.T, Y.reshape(-1)).reshape(3, 3))