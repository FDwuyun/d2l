import torch
import torchvision
import d2l_13 as d2l
from d2l_13 import *
from torch import nn
from torch.nn import functional as F
from torchvision import models

d2l.set_figsize()
content_img = d2l.Image.open(script_dir + '/../img/content.jpg')
# d2l.plt.imshow(content_img)


style_img = d2l.Image.open(script_dir + '/../img/style.jpg')
# d2l.plt.imshow(style_img)


rgb_mean = torch.tensor([0.485, 0.456, 0.406])
rgb_std = torch.tensor([0.229, 0.224, 0.225])

def preprocess(img, image_shape):
    """图片变为tensor。预处理函数`preprocess`对输入图像在RGB三个通道分别做标准化，并将结果变换成卷积神经网络接受的输入格式。"""
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize(image_shape),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=rgb_mean, std=rgb_std)])
    return transforms(img).unsqueeze(0)

def postprocess(img):
    """tensor变为图片。后处理函数`postprocess`则将输出图像中的像素值还原回标准化之前的值。
    由于图像打印函数要求每个像素的浮点数值在0到1之间，我们对小于0和大于1的值分别取0和1。
    """
    img = img[0].to(rgb_std.device)
    img = torch.clamp(img.permute(1, 2, 0) * rgb_std + rgb_mean, 0, 1)
    return torchvision.transforms.ToPILImage()(img.permute(2, 0, 1))

# 使用基于ImageNet数据集预训练的VGG-19模型来抽取图像特征。
pretrained_net = torchvision.models.vgg19(weights = models.VGG19_Weights.IMAGENET1K_V1)
# X = torch.rand((1, 3, 320, 480))
# summary(pretrained_net, input_data=X)
style_layers, content_layers = [0, 5, 10, 19, 28], [25] # 越小越靠近输入（局部样式），越大越靠近输出（全局样式）

# print(max(content_layers + style_layers) + 1)
# 丢弃最大层之后的层
net = nn.Sequential( *[
    pretrained_net.features[i] for i in range(max(content_layers + style_layers) + 1)
])

def extract_features(X, content_layers, style_layers):
    """输入X，获取内容层和样式层的输出"""
    contents = []
    styles = []
    for i in range(len(net)):
        X = net[i](X)
        if i in style_layers:
            styles.append(X)
        if i in content_layers:
            contents.append(X)
    return contents, styles

# 因为训练的时候无需改变预训练的VGG的参数，所以我们可以在训练开始之前抽取内容特征和样式特征
# 由于合成图像是样式迁移时所需要迭代的模型参数，我们只能在训练过程中调用extract_features函数来抽取合成图像的内容特征和样式特征
def get_contents(image_shape, device):
    """对内容图像抽取内容特征"""
    content_X = preprocess(content_img, image_shape).to(device)
    contents_Y, _ = extract_features(content_X, content_layers, style_layers)
    return content_X, contents_Y

def get_styles(image_shape, device):
    """对样式图像抽取样式特征"""
    style_X = preprocess(style_img, image_shape).to(device)
    _, styles_Y = extract_features(style_X, content_layers, style_layers)
    return style_X, styles_Y

# 4. 定义损失函数 样式迁移的损失函数。它由内容损失、样式损失和总变差损失3部分组成。

def content_loss(Y_hat, Y):
    """合成图像与内容图像在内容特征上的差异。
    与线性回归中的损失函数类似，内容损失通过平方误差函数衡量合成图像与内容图像在内容特征上的差异。
    平方误差函数的两个输入均为extract_features函数计算所得到的内容层的输出。
    """
    # 我们从动态计算梯度的树中分离目标：
    # 这是一个规定的值，而不是一个变量。
    return torch.square(Y_hat - Y.detach()).mean()

def gram(X):
    """gram函数将格拉姆矩阵除以了矩阵中元素的个数，即chw。
    当h w的值较大时，格拉姆矩阵中的元素容易出现较大的值。
    此外，格拉姆矩阵的高和宽皆为通道数c。为了让样式损失不受这些值的大小影响，
    """
    num_channels, hw = X.shape[1], X.numel() // X.shape[1] # channel为通道数，hw为高宽乘积
    X = X.reshape((num_channels, hw))
    return torch.matmul(X, X.T) / (num_channels * hw)

def style_loss(Y_hat, gram_Y):
    """自然地，样式损失的平方误差函数的两个格拉姆矩阵输入分别基于合成图像与样式图像的样式层输出。
    这里假设基于样式图像的格拉姆矩阵gram_Y已经预先计算好了。
    """
    return torch.square(gram(Y_hat) - gram_Y.detach()).mean()

def tv_loss(Y_hat):
    """总变差损失，学到的合成图像里面有大量高频噪点，即有特别亮或者特别暗的颗粒像素。一种常见的降噪方法是总变差降噪。
    TV降噪，使得邻近像素值类似
    """
    return 0.5 * (torch.abs(Y_hat[:, :, 1:, :] - Y_hat[:, :, :-1, :]).mean() +
                  torch.abs(Y_hat[:, :, :, 1:] - Y_hat[:, :, :, :-1]).mean())


content_weight, style_weight, tv_weight = 10, 1e12, 1

def compute_loss(X, contents_Y_hat, styles_Y_hat, contents_Y, styles_Y_gram):
    """风格转移的损失函数是内容损失、风格损失和总变化损失的加权和通过调节这些权值超参数，
    我们可以权衡合成图像在保留内容、迁移样式以及降噪三方面的相对重要性。
    """
    contents_l = [
        content_loss(Y_hat, Y) * content_weight
        for Y_hat, Y in zip(contents_Y_hat, contents_Y)]
    styles_l = [
        style_loss(Y_hat, Y) * style_weight
        for Y_hat, Y in zip(styles_Y_hat, styles_Y_gram)]
    tv_l = tv_loss(X) * tv_weight
    # 对所有损失求和
    l = sum(10 * styles_l + contents_l + [tv_l])
    return contents_l, styles_l, tv_l, l

class SynthesizedImage(nn.Module):
    def __init__(self, img_shape):
        super(SynthesizedImage, self).__init__()
        self.weight = nn.Parameter(torch.rand(*img_shape))

    def forward(self):
        return self.weight

def get_inits(X, device, lr, styles_Y):
    """
    该函数创建了合成图像的模型实例，并将其初始化为图像 `X` 。
    样式图像在各个样式层的格拉姆矩阵 `styles_Y_gram` 将在训练前预先计算好。
    """
    # X是内容图像的预处理结果
    gen_img = SynthesizedImage(X.shape).to(device)
    # 将初始化的weight参数改为已有的图像X的参数（即像素）
    gen_img.weight.data.copy_(X.data)
    # 定义优化器
    trainer = torch.optim.Adam(gen_img.parameters(), lr=lr)
    # 对各风格特征图计算其格拉姆矩阵，并依次存于列表中
    styles_Y_gram = [gram(Y) for Y in styles_Y]
    # !!!gen_img()!!!括号
    return gen_img(), styles_Y_gram, trainer

def train(X, contents_Y, styles_Y, device, lr, num_epochs, lr_decay_epoch):
    timer2 = Timer()
    timer2.start()
    
    # X是初始化的合成图像，style_Y_gram是原始风格图像的格拉姆矩阵列表
    X, styles_Y_gram, trainer = get_inits(X, device, lr, styles_Y)
    # 定义学习率下降调节器
    scheduler = torch.optim.lr_scheduler.StepLR(trainer, lr_decay_epoch, 0.8) # 依次降低学习率
    # animator = d2l.Animator(xlabel='epoch', ylabel='loss',
    #                         xlim=[10, num_epochs],
    #                         legend=['content', 'style', 'TV'], ncols=2)
    timer = Timer()
    
    for epoch in range(num_epochs):
        timer.start()

        trainer.zero_grad()
        # Y_hat是用合成图像计算出的特征图
        contents_Y_hat, styles_Y_hat = extract_features(
            X, content_layers, style_layers)
        contents_l, styles_l, tv_l, l = compute_loss(X, contents_Y_hat,
                                                     styles_Y_hat, contents_Y,
                                                     styles_Y_gram)
        # 反向传播误差（计算l对合成图像像素矩阵的导数，因为l的唯一自变量是合成图像像素矩阵）
        l.backward()
        # 更新一次合成图像的像素参数
        trainer.step()
        # 更新学习率超参数
        scheduler.step()
        timer.stop()
        if (epoch + 1) % 50 == 0:
            # animator.axes[1].imshow(postprocess(X))
            # animator.add(
            #     epoch + 1,
            #     [float(sum(contents_l)),
            #      float(sum(styles_l)),
            #      float(tv_l)])
            print(f"[{get_datetime()}][epoch: {epoch+1}] {timer.sum():.1f} sec,", 
              f'contents_l {sum(contents_l).item():.9f}, styles_l {sum(styles_l).item():.9f}, 总变差损失 {tv_l.item():.9f}')
        
    timer2.stop()
    print(f"[Total {num_epochs} epochs] {timer2.sum():.1f} sec")
    return X


# 现在我们[**训练模型**]：首先将内容图像和样式图像的高和宽分别调整为300和450像素，用内容图像来初始化合成图像。
device, image_shape = d2l.try_gpu(), (900, 1350)
net = net.to(device)
# 计算内容图像的预处理结果（因为我们将内容图像作为合成图像的初始化图像作为网络的初始输入）和抽取到的内容特征
content_X, contents_Y = get_contents(image_shape, device)
# 计算风格图像抽取到的风格特征
_, styles_Y = get_styles(image_shape, device)
output = train(content_X, contents_Y, styles_Y, device, 0.1, 50000, 200)

plt.imshow(postprocess(output))
plt.axis('off')
plt.tight_layout()
plt.savefig(script_dir + "/CompositeImage_2.png")





