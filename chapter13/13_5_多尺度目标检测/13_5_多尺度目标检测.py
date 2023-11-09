import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models,transforms,datasets
from torch.utils import data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = os.getcwd()

import math
import matplotlib
matplotlib.use("Agg")  # 这一句一定要放在下面这句的前面
from matplotlib import pyplot as plt

img = plt.imread(script_dir + '/../img/catdog.jpg')
h, w = img.shape[:2]

def set_figsize(figsize=(7, 5)):
    """设置matplotlib的图表大小"""
    plt.rcParams["figure.figsize"] = figsize

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框，生成的是归一化之后的，需要乘以h,w才是真实的锚框;输出(1, h*w, 4)"""
    # in_height, in_width = data.shape[-2:]
    in_height = data.shape[-2]
    in_width = data.shape[-1]

    device, num_sizes, num_ratios = data.device, len(sizes), len(ratios)
    boxes_per_pixel = (num_sizes + num_ratios -1)
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)

    # 为了将锚点移动到像素的中心，需要设置偏移量
    # 因为一个像素的高为1且宽为1，我们选择偏移中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height # 在y轴上缩放步长
    steps_w = 1.0 / in_width # 在x轴上缩放步长

    # 生成锚框的所有中心点
    center_h = (torch.arange(in_height, device=device) + offset_h) * steps_h
    center_w = (torch.arange(in_width, device=device) + offset_w) * steps_w
    # 其中第一个输出张量填充第一个输入张量中的元素，各行元素相同；第二个输出张量填充第二个输入张量中的元素，各列元素相同
    shift_y, shift_x = torch.meshgrid(center_h, center_w, indexing='ij')
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1)

    # 生成“boxes_per_pixel”个高和宽，
    # 之后用于创建锚框的四角坐标(xmin, ymin, xmax, ymax)
    # numel(w) = boxes_per_pixel = (num_sizes + num_ratios -1) , 只考虑包含s1或r1的组合: (s1, r1), (s1, r2), ..., (s1, rm), (s2, r1), (s3, r1), ..., (sn, r1)
    # w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:])))  # 处理矩形输入
    # h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))
    
    w = torch.cat((sizes[0] * torch.sqrt(ratio_tensor), size_tensor[1:] * torch.sqrt(ratio_tensor[0]))) * math.sqrt(in_height / in_width)
    h = torch.cat((sizes[0] / torch.sqrt(ratio_tensor), size_tensor[1:] / torch.sqrt(ratio_tensor[0]))) * math.sqrt(in_width / in_height)
    # 除以2来获得半高和半宽，repeat_interleave()：在原有的tensor上，按每一个tensor复制。repeat()：根据原有的tensor复制n个，然后拼接在一起。
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有"boxes_per_pixel"个锚框
    # 所以生成含所有锚框中心的网络，重复了"boxes_per_pixel"次
    # repeat_interleave()：在原有的tensor上，按每一个tensor复制。repeat()：根据原有的tensor复制n个，然后拼接在一起。
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    # unsqueeze 函数用于在指定维度上给张量增加一个维度，从而改变张量的形状。在这里，unsqueeze(0) 表示在维度 0 上增加一个维度，即在张量的最外层添加一个维度。
    return output.unsqueeze(0)

def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )

def show_bboxes(axes, bboxes, labels=None, colors=None):
    """显示所有边界框"""
    # 该函数用于将对象转换为列表形式
    def _make_list(obj, default_values=None):
        if obj is None:
            obj = default_values
        elif not isinstance(obj, (list, tuple)):
            obj = [obj]
        return obj
    
    labels = _make_list(labels)
    colors = _make_list(colors, ['b', 'g', 'r', 'm', 'c'])
    for i, bbox in enumerate(bboxes):
        color = colors[i % len(colors)]
        rect = bbox_to_rect(bbox.detach().numpy(), color)
        axes.add_patch(rect)
        if labels and len(labels) > i:
            text_color = 'k' if color == 'w' else 'w'
            #  lw（linewidth）设置为 0 表示不显示边框线条
            axes.text(rect.xy[0], rect.xy[1], labels[i],
                      va='center', ha='center', fontsize=9, color=text_color,
                      bbox=dict(facecolor=color, lw=0))

def display_anchors(fmap_w, fmap_h, s):
    set_figsize()
    # 前两个维度上的值不影响输出
    fmap = torch.zeros((1, 10, fmap_h, fmap_w))
    # sizes^2 = h1*w1/h*w, ratios = w1/ h1 
    anchors = multibox_prior(fmap, sizes=s, ratios=[1, 2, 0.5])
    bbox_scale = torch.tensor((w, h, w, h))
    show_bboxes(plt.imshow(img).axes, anchors[0] * bbox_scale)

display_anchors(fmap_w=1, fmap_h=1, s=[0.8])
plt.savefig(script_dir + "/catdog_4x4.png")
