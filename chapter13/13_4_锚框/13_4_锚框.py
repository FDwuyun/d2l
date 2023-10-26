import torch
from torch import nn
from torch.nn import functional as F
import torchvision
from torchvision import models,transforms,datasets
from torch.utils import data
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
# script_dir = os.getcwd()

# 精简输出精度
torch.set_printoptions(2)

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框"""
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
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]), sizes[0] * torch.sqrt(ratio_tensor[1:]))) * in_height / in_width # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]), sizes[0] / torch.sqrt(ratio_tensor[1:])))
    # 除以2来获得半高和半宽，repeat_interleave()：在原有的tensor上，按每一个tensor复制。repeat()：根据原有的tensor复制n个，然后拼接在一起。
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(in_height * in_width, 1) / 2

    # 每个中心点都将有"boxes_per_pixel"个锚框
    # 所以生成含所有锚框中心的网络，重复了"boxes_per_pixel"次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y], dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
