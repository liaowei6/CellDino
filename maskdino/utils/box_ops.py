# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
import math
from torchvision.ops.boxes import box_area
from mmcv.ops import nms_rotated

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def obox_xyxy_to_cxcywht(x, norm_angle = False):  #norm_angle: 角度是否需要归一化
    x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = x.unbind(-1)    
    if norm_angle:
        b = [(x_1 + x_2 +x_3 +x_4) / 4, (y_1 + y_2 + y_3 + y_4) / 4,
        torch.sqrt(torch.square(x_2 - x_3) + torch.square(y_2 - y_3)), 
        torch.sqrt(torch.square(x_2 - x_1) + torch.square(y_2 - y_1)),
        torch.atan((x_2 - x_3) / (y_3 - y_2)) / math.pi * 2]
    else:
        b = [(x_1 + x_2 +x_3 +x_4) / 4, (y_1 + y_2 + y_3 + y_4) / 4,
            torch.sqrt(torch.square(x_2 - x_3) + torch.square(y_2 - y_3)), 
            torch.sqrt(torch.square(x_2 - x_1) + torch.square(y_2 - y_1)),
            torch.atan((x_2 - x_3) / (y_3 - y_2))]
    return torch.stack(b, dim=-1)

def obox_cxcywht_to_xyxy(x, norm_angle = False):  #norm_angle: 角度是否归一化了
    x_c, y_c, w, h, t = x.unbind(-1)
    if norm_angle:
        t = t * math.pi / 2
    else:
        t = t / 90 * math.pi / 2
    x_1 = (x_c - w * 0.5 * torch.cos(-t) - h * 0.5 * torch.sin(-t))
    y_1 = (y_c - w * 0.5 * torch.sin(-t) + h * 0.5 * torch.cos(-t))
    x_2 = (x_c + w * 0.5 * torch.cos(-t) - h * 0.5 * torch.sin(-t))
    y_2 = (y_c + w * 0.5 * torch.sin(-t) + h * 0.5 * torch.cos(-t))
    x_3 = (x_c + w * 0.5 * torch.cos(-t) + h * 0.5 * torch.sin(-t))
    y_3 = (y_c + w * 0.5 * torch.sin(-t) - h * 0.5 * torch.cos(-t))
    x_4 = (x_c - w * 0.5 * torch.cos(-t) + h * 0.5 * torch.sin(-t))
    y_4 = (y_c - w * 0.5 * torch.sin(-t) - h * 0.5 * torch.cos(-t))

    b = [torch.cat((x_1.unsqueeze(1), y_1.unsqueeze(1)), dim = 1).unsqueeze(1),
         torch.cat((x_2.unsqueeze(1), y_2.unsqueeze(1)), dim = 1).unsqueeze(1),
         torch.cat((x_3.unsqueeze(1), y_3.unsqueeze(1)), dim = 1).unsqueeze(1),
         torch.cat((x_4.unsqueeze(1), y_4.unsqueeze(1)), dim = 1).unsqueeze(1),  
        ]
    return torch.cat(b, dim=1)


def scale_obbox(x, scale, norm_angle = True):
    x = x.unsqueeze(-2)
    scale = scale.unsqueeze(-2)
    x_c, y_c, w, h, t = x.unbind(-1)
    scale_x, scale_y = scale.unbind(-1)
    theta = t * math.pi / 180.0
    if norm_angle:
        theta = theta * 90
    x_c = x_c * scale_x
    y_c = y_c * scale_y
    c = torch.cos(theta)
    s = torch.sin(theta)

    w = w * torch.sqrt((scale_x * c) ** 2 + (scale_y * s) ** 2)

    # h(new) = |F(new) - O| * 2
    #        = sqrt[(sfx * s * h / 2)^2 + (sfy * c * h / 2)^2] * 2
    #        = sqrt[(sfx * s)^2 + (sfy * c)^2] * h
    # i.e., scale_factor_h = sqrt[(sfx * s)^2 + (sfy * c)^2]
    #
    # For example,
    # when angle = 0 or 180, |c| = 1, s = 0, scale_factor_h == scale_factor_y;
    # when |angle| = 90, c = 0, |s| = 1, scale_factor_h == scale_factor_x
    h  = h * torch.sqrt((scale_x * s) ** 2 + (scale_y * c) ** 2)

    # The angle is the rotation angle from y-axis in image space to the height
    # vector (top->down in the box's local coordinate system) of the box in CCW.
    #
    # angle(new) = angle_yOx(O - F(new))
    #            = angle_yOx( (sfx * s * h / 2, sfy * c * h / 2) )
    #            = atan2(sfx * s * h / 2, sfy * c * h / 2)
    #            = atan2(sfx * s, sfy * c)
    #
    # For example,
    # when sfx == sfy, angle(new) == atan2(s, c) == angle(old)
    t = torch.atan2(scale_x * s, scale_y * c) * 180 / math.pi
    if norm_angle:
        t = t / 90

    return torch.stack([x_c, y_c, w, h, t], dim=-1).squeeze(-2)
# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)


    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-6)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-6)



# modified from torchvision to also return the union
def box_iou_pairwise(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N,2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N,2]

    wh = (rb - lt).clamp(min=0)  # [N,2]
    inter = wh[:, 0] * wh[:, 1]  # [N]

    union = area1 + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou_pairwise(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    Input:
        - boxes1, boxes2: N,4
    Output:
        - giou: N, 4
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    assert boxes1.shape == boxes2.shape
    iou, union = box_iou_pairwise(boxes1, boxes2) # N, 4

    lt = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,2]
    area = wh[:, 0] * wh[:, 1]

    return iou - (area - union) / area

def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float, device=masks.device)
    x = torch.arange(0, w, dtype=torch.float, device=masks.device)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def multiclass_nms_rotated(multi_bboxes,
                           multi_scores,
                           score_thr,
                           nms_thr,
                           max_num=-1,
                           score_factors=None,
                           return_inds=False):
    """NMS for multi-class bboxes.

    Args:
        multi_bboxes (torch.Tensor): shape (n, #class*5) or (n, 5)
        multi_scores (torch.Tensor): shape (n, #class), where the last column
            contains scores of the background class, but this will be ignored.
        score_thr (float): bbox threshold, bboxes with scores lower than it
            will not be considered.
        nms (float): Config of NMS.
        max_num (int, optional): if there are more than max_num bboxes after
            NMS, only top max_num will be kept. Default to -1.
        score_factors (Tensor, optional): The factors multiplied to scores
            before applying NMS. Default to None.
        return_inds (bool, optional): Whether return the indices of kept
            bboxes. Default to False.

    Returns:
        tuple (dets, labels, indices (optional)): tensors of shape (k, 5), \
        (k), and (k). Dets are boxes with scores. Labels are 0-based.
    """
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 5:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 5)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 5)
    scores = multi_scores[:, :-1]

    labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
    labels = labels.view(1, -1).expand_as(scores)
    bboxes = bboxes.reshape(-1, 5)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)

    # remove low scoring boxes
    valid_mask = scores > score_thr
    if score_factors is not None:
        # expand the shape to match original shape of score
        score_factors = score_factors.view(-1, 1).expand(
            multi_scores.size(0), num_classes)
        score_factors = score_factors.reshape(-1)
        scores = scores * score_factors

    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]

    if bboxes.numel() == 0:
        dets = torch.cat([bboxes, scores[:, None]], -1)
        if return_inds:
            return dets, scores, labels, inds
        else:
            return dets, labels

    # Strictly, the maximum coordinates of the rotating box (x,y,w,h,a)
    # should be calculated by polygon coordinates.
    # But the conversion from rbbox to polygon will slow down the speed.
    # So we use max(x,y) + max(w,h) as max coordinate
    # which is larger than polygon max coordinate
    # max(x1, y1, x2, y2,x3, y3, x4, y4)
    max_coordinate = bboxes[:, :2].max() + bboxes[:, 2:4].max()
    offsets = labels.to(bboxes) * (max_coordinate + 1)
    if bboxes.size(-1) == 5:
        bboxes_for_nms = bboxes.clone()
        bboxes_for_nms[:, :2] = bboxes_for_nms[:, :2] + offsets[:, None]
    else:
        bboxes_for_nms = bboxes + offsets[:, None]
    _, keep = nms_rotated(bboxes_for_nms, scores, nms_thr)

    if max_num > 0:
        keep = keep[:max_num]

    bboxes = bboxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if return_inds:
        return bboxes, scores, labels, inds[keep]
    else:
        return bboxes, labels

if __name__ == '__main__':
    x = torch.rand(5, 4)
    y = torch.rand(3, 4)
    iou, union = box_iou(x, y)