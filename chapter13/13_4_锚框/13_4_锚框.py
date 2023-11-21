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
# torch.set_printoptions(2)

import math

def multibox_prior(data, sizes, ratios):
    """生成以每个像素为中心具有不同形状的锚框，生成的是归一化之后的，需要乘以h,w才是真实的锚框"""
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

import matplotlib
matplotlib.use("Agg")  # 这一句一定要放在下面这句的前面
from matplotlib import pyplot as plt

img = plt.imread(script_dir + '/../img/catdog.jpg')
h, w = img.shape[:2]

# print(h, w)
# X = torch.rand(size=(1, 3, h, w))
# Y = multibox_prior(X, sizes=[0.75, 0.5, 0.25], ratios=[1, 2, 0.5])
# print(Y.shape)

# (图像高度,图像宽度,以同一像素为中心的锚框的数量(n+m-1), 边界框(左上x,左上y,右下x,右下y))
# boxes = Y.reshape(h, w, 5, 4)
# print(Y.shape)
# print(boxes[250, 250, :, :])

def bbox_to_rect(bbox, color):
    # 将边界框(左上x,左上y,右下x,右下y)格式转换成matplotlib格式：
    # ((左上x,左上y),宽,高)
    return plt.Rectangle(
        xy=(bbox[0], bbox[1]), width=bbox[2]-bbox[0], height=bbox[3]-bbox[1],
        fill=False, edgecolor=color, linewidth=2
    )

def set_figsize(figsize=(14, 10)):
    """设置matplotlib的图表大小"""
    plt.rcParams["figure.figsize"] = figsize

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
            
# set_figsize()
bbox_scale = torch.tensor((w, h, w, h))
# fig = plt.imshow(img)
# print(boxes[250, 250, :, :] * bbox_scale)
# show_bboxes(fig.axes, boxes[250, 250, :, :] * bbox_scale,
#             ['s=0.75, r=1', 's=0.75, r=2', 's=0.75, r=0.5', 's=0.5, r=1', 's=0.25, r=1'])
# plt.savefig(script_dir + f"/catdog_bboxes.jpg")

def box_iou(boxes1, boxes2):
    """
    输入：boxes1:[boxes1数量,(左上x,左上y,右下x,右下y)],boxes2:[boxes2数量,4]，boxes1是锚框，boxes2是真实gt边界框
    输出：交并比[boxes1数量,boxes2数量]
    """
    #定义lambda表达式，计算矩形框的面积
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))

    #计算锚框（anchor box）的面积,[boxes1数量]
    areas1 = box_area(boxes1)
    #计算真实边界框gt bounding box的面积,[boxes2数量]
    areas2 = box_area(boxes2)

    # 相交区域inter_upperlefts, inter_lowerrights, inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    # None 的作用是在第二个维度插入一个新的轴，即将数组变为 (boxes1的数量, 1, 4) 的形状。
    inter_lowerrights = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    # 得到交集的宽和高，[boxes1数量,boxes2数量,2],clamp限制输出最小为0(min=0)
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    # 得到交集的面积[boxes1数量,boxes2数量]
    inter_areas = inters[:, :, 0] * inters[:, :, 1]
    # 并集的面积等于两个边界框列表中所有边界框的面积之和减去交集的面积。
    # 得到并集的面积[boxes1数量,boxes2数量],此处又用到了广播机制
    union_areas = areas1[:, None] + areas2 - inter_areas
    return inter_areas / union_areas

def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    """将最接近的真实边界框分配给锚框"""
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    # 位于第i行和第j列的元素x_ij是锚框i和真实边界框j的IoU
    jaccard = box_iou(anchors, ground_truth)
    # 对于每个锚框，分配的真实边界框的张量
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    # 遍历交并比矩阵每一行，寻找每一行最大值，根据阈值，决定是否分配真实边界框
    max_ious, indices = torch.max(jaccard, dim=1)
    anc_i = torch.nonzero(max_ious >= iou_threshold).reshape(-1)
    box_j = indices[max_ious >= iou_threshold]
    anchors_bbox_map[anc_i] = box_j
    # 在jaccard交并比矩阵中，寻找每个真实框的最大值
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard)
        box_idx = (max_idx % num_gt_boxes).long()
        anc_idx = (max_idx / num_gt_boxes).long()
        anchors_bbox_map[anc_idx] = box_idx
        jaccard[:, box_idx] = col_discard
        jaccard[anc_idx, :] = row_discard
    return anchors_bbox_map

def box_corner_to_center(boxes):
    """从（左上，右下）转换到（中间，宽度，高度）"""
    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    w = x2 - x1
    h = y2 - y1
    boxes = torch.stack((cx, cy, w, h), axis=-1)
    return boxes

def offset_boxes(anchors, assigned_bb, eps=1e-6):
    """对锚框偏移量的转换"""
    c_anc = box_corner_to_center(anchors)
    c_assigned_bb = box_corner_to_center(assigned_bb)
    offset_xy = 10 * (c_assigned_bb[:, :2] - c_anc[:, :2]) / c_anc[:, 2:]
    offset_wh = 5 * torch.log(eps + c_assigned_bb[:, 2:] / c_anc[:, 2:])
    offset = torch.cat([offset_xy, offset_wh], axis=1)
    return offset

def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    # .squeeze(0) 方法将删除第一个维度为 1 的维度。这种操作通常用于减少张量的尺寸，
    # 如果第一个维度确实是长度为 1 的维度，那么它将被压缩。如果第一个维度不是长度为 1 的维度，则不会进行任何操作。
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    for i in range(batch_size):
        # labels[batch_size, num_gt_boxes, [?, 左上x, 左上y, 右下x, 右下y]]
        label = labels[i, :, :]
        # label[num_gt_boxes, [?, 左上x, 左上y, 右下x, 右下y]]
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        # anchors_bbox_map: [], size = num_anchors = anchors.shape[0]
        # 扩展为四列
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1)).repeat(1, 4)
        # 将类标签和分配的边界框坐标初始化为零
        class_labels = torch.zeros(num_anchors, dtype=torch.long, device=device)
        # class_labels.shape : [num_anchors]
        assigned_bb = torch.zeros((num_anchors, 4), dtype=torch.float32, device=device)
        # assigned_bb.shape : [num_anchors, 4]

        # 使用真实边界框来标记锚框的类别。
        # 如果一个锚框没有被分配，标记其为背景（值为零）
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        bb_idx = anchors_bbox_map[indices_true]
        
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        assigned_bb[indices_true] = label[bb_idx, 1:]
        # 偏移量转换
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)


ground_truth = torch.tensor(
    [
        [0, 0.1, 0.08, 0.52, 0.92],
        [1, 0.55, 0.2, 0.9, 0.88]
    ]
)

anchors = torch.tensor(
    [
        [0, 0.1, 0.2, 0.3],
        [0.15, 0.2, 0.4, 0.4],
        [0.63, 0.05, 0.88, 0.98], 
        [0.66, 0.45, 0.8, 0.8],
        [0.57, 0.3, 0.92, 0.9]
    ]
)

# fig = plt.imshow(img)
# show_bboxes(fig.axes, ground_truth[:, 1:] * bbox_scale, ['dog', 'cat'], 'k')
# show_bboxes(fig.axes, anchors * bbox_scale, ['0', '1', '2', '3', '4'])
# plt.savefig(script_dir + f"/catdog_bboxes.jpg")

labels = multibox_target(anchors.unsqueeze(dim=0), ground_truth.unsqueeze(dim=0))
# print(labels)

def box_center_to_corner(boxes):
    """从（中间，宽度，高度）转换到（左上，右下）"""
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - 0.5 * w
    y1 = cy - 0.5 * h
    x2 = cx + 0.5 * w
    y2 = cy + 0.5 * h
    boxes = torch.stack((x1, y1, x2, y2), axis=-1)
    return boxes

def offset_inverse(anchors, offset_preds):
    """根据带有预测偏移量的锚框来预测边界框"""
    # 从（左上，右下）转换到（中间，宽度，高度）
    anc = box_corner_to_center(anchors)
    pred_bbox_xy = (offset_preds[:, :2] * anc[:, 2:] / 10) + anc[:, :2]
    pred_bbox_wh = torch.exp(offset_preds[:, 2:] / 5) * anc[:, 2:]
    pred_bbox = torch.cat((pred_bbox_xy, pred_bbox_wh), axis=1)
    predicted_bbox = box_center_to_corner(pred_bbox)
    return predicted_bbox

def nms(boxes, scores, iou_threshold):
    """
    非极大值抑制（non-maximum suppression，NMS），对预测边界框的置信度进行排序
    ，保留预测边界框的指标
    """
    # 按照最后一个维度进行降序排序, 返回的是索引值
    B = torch.argsort(scores, dim=-1, descending=True)
    keep = []  # 保留预测边界框的指标
    while B.numel() > 0:
        # 从B中选取置信度最高的预测边界框B[0]作为基准
        i = B[0]
        keep.append(i)
        # 如果只剩最后一个元素
        if B.numel() == 1: break
        iou = box_iou(
            boxes[i, :].reshape(-1, 4), 
            boxes[B[1:], :].reshape(-1, 4)
        ).reshape(-1)
        #  函数用于找到所有为 True 的元素的索引
        inds = torch.nonzero(iou <= iou_threshold).reshape(-1)
        B = B[inds + 1]
    return torch.tensor(keep, device=boxes.device)

def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                       pos_threshold=0.009999999):
    """使用非极大值抑制来预测边界框"""
    device, batch_size = cls_probs.device, cls_probs.shape[0]
    anchors = anchors.squeeze(0)
    num_classes, num_anchors = cls_probs.shape[1], cls_probs.shape[2]
    out = []
    for i in range(batch_size):
        cls_probs, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
        conf, class_id = torch.max(cls_probs[1:], 0)
        predicted_bb = offset_inverse(anchors, offset_pred)
        keep = nms(predicted_bb, conf, nms_threshold)

        # 找到所有的non_keep索引， 并将类设置为背景
        all_idx = torch.arange(num_anchors, dtype=torch.long, device=device)
        combined = torch.cat((keep, all_idx))
        uniques, counts = combined.unique(return_counts=True)
        non_keep = uniques[counts == 1]
        all_id_sorted = torch.cat((keep, non_keep))
        class_id[non_keep] = -1
        class_id = class_id[all_id_sorted]
        conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
        # pos_threshold是一个用于非常背景预测的阈值
        below_min_idx = (conf < pos_threshold)
        conf[below_min_idx] = 1 - conf[below_min_idx]
        pred_info = torch.cat(
            (class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb),
            dim=1
        )
        out.append(pred_info)
    return torch.stack(out)
    
anchors = torch.tensor(
    [
        [0.1, 0.08, 0.52, 0.92],
        [0.08, 0.2, 0.56, 0.95],
        [0.15, 0.3, 0.62, 0.91],
        [0.55, 0.2, 0.9, 0.88]
    ]
)
offset_preds = torch.tensor([0] * anchors.numel())
cls_probs = torch.tensor(
    [
        [0] * 4,  # 背景的预测概率
        [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
        [0.1, 0.2, 0.3, 0.9]
    ]
)  # 猫的预测概率

# fig = plt.imshow(img)
# show_bboxes(fig.axes, anchors * bbox_scale,
#             ['dog=0.9', 'dog=0.8', 'dog=0.7', 'cat=0.9'])

output = multibox_detection(cls_probs.unsqueeze(dim=0),
                            offset_preds.unsqueeze(dim=0),
                            anchors.unsqueeze(dim=0),
                            nms_threshold=0.5)

fig = plt.imshow(img)
for i in output[0].detach().numpy():
    if i[0] == -1:
        continue
    label = ('dog=', 'cat=')[int(i[0])] + str(i[1])
    show_bboxes(fig.axes, [torch.tensor(i[2:]) * bbox_scale], label)

plt.savefig(script_dir + "/catdog_predicted.jpg")
