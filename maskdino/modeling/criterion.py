# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from DINO https://github.com/IDEA-Research/DINO by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------
"""
MaskFormer criterion.
"""
import logging

import torch
import torch.distributions as dist
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn
import numpy as np
import math
from detectron2.structures import BitMasks
from detectron2.utils.comm import get_world_size
from detectron2.projects.point_rend.point_features import (
    get_uncertain_point_coords_with_randomness,
    point_sample,
)
from pycocotools import mask as coco_mask
from ..utils import box_ops
from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list
from maskdino.utils import box_ops
from mmcv.ops import box_iou_rotated
from .loss_levelset import LevelsetLoss, LCM

def convert_coco_poly_to_mask(segmentations, height, width):
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks

def get_point_coords_with_gt_mask(gaussian_matrix, num_points):
    mean, cov = xy_wh_r_2_xy_sigma(gaussian_matrix.to(torch.float64))
    gaussian = dist.MultivariateNormal(mean, cov)
    samples = gaussian.sample((num_points,)).transpose(0,1)
    samples = torch.clamp(samples, 0, 1)
    return samples

def _scale_target(targets_, scaled_size=(96, 96)):
    """ scale the targets to the scales size.

    """

    if targets_.dim() == 3:
        targets = targets_.unsqueeze(1)
    else:
        targets = targets_
    targets = F.interpolate(targets, size=scaled_size, mode='bilinear', align_corners=False)

    return targets

@torch.no_grad()
def cost_matrix_compute(outputs, targets, cost=["cls", "obox"]):
    """More memory-friendly matching. Change cost to compute only certain loss in matching"""
    bs, num_queries = outputs["pred_logits"].shape[:2]

    # Iterate through batch size
    # ious = []
    cost_matrix = []
    for b in range(bs):
        out_bbox = outputs["pred_boxes"][b].to(torch.float64)
        tgt_bbox = targets[b]["boxes"].to(torch.float64)
        angle_scale = torch.tensor([1.0, 1.0, 1.0, 1.0, 90.0], device= out_bbox.device)
        angle_convert = torch.tensor([1.0, 1.0, 1.0, 1.0, math.pi / 180.0], device= out_bbox.device)
        cost_bbox = torch.cdist(out_bbox, tgt_bbox / angle_scale, p=1)
        pre = out_bbox * angle_scale * angle_convert
        tgt = tgt_bbox * angle_convert
        cost_giou = -box_iou_rotated(pre.float(), tgt.float())
        #ious.append(-cost_giou)
        out_prob = outputs["pred_logits"][b].sigmoid()  # [num_queries, num_classes]
        tgt_ids = targets[b]["labels"]
        # focal loss
        alpha = 0.25
        gamma = 2.0
        neg_cost_class = (1 - alpha) * (out_prob ** gamma) * (-(1 - out_prob + 1e-8).log())
        pos_cost_class = alpha * ((1 - out_prob) ** gamma) * (-(out_prob + 1e-8).log())
        cost_class = pos_cost_class[:, tgt_ids] - neg_cost_class[:, tgt_ids]

        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        # cost_class = -out_prob[:, tgt_ids]

        C = ( 4 * cost_class
            + 5 * cost_bbox
        )
        cost_matrix.append(C)

    return cost_matrix


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss


    return loss.mean(1).sum() / num_boxes


def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """
    loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")

    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule

def xy_wh_r_2_xy_sigma(xywhr):
    """Convert oriented bounding box to 2-D Gaussian distribution.

    Args:
        xywhr (torch.Tensor): rbboxes with shape (N, 5).

    Returns:
        xy (torch.Tensor): center point of 2-D Gaussian distribution
            with shape (N, 2).
        sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
            with shape (N, 2, 2).
    """
    _shape = xywhr.shape
    assert _shape[-1] == 5
    xy = xywhr[..., :2]
    wh = xywhr[..., 2:4].clamp(min=1e-7, max=1e7).reshape(-1, 2)
    r = xywhr[..., 4] * math.pi / 180.0 
    cos_r = torch.cos(r)
    sin_r = torch.sin(r)
    R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
    S = 0.5 * torch.diag_embed(wh)

    sigma = R.bmm(S.square()).bmm(R.permute(0, 2,
                                            1)).reshape(_shape[:-1] + (2, 2))

    return xy, sigma


def kfiou_loss(pred,
               target,
               fun=None,
               beta=1.0 / 9.0,
               eps=1e-6):
    """Kalman filter IoU loss.

    Args:
        pred (torch.Tensor): Predicted bboxes.  归一化
        target (torch.Tensor): Corresponding gt bboxes. 归一化
        pred_decode (torch.Tensor): Predicted decode bboxes.
        targets_decode (torch.Tensor): Corresponding gt decode bboxes.
        fun (str): The function applied to distance. Defaults to None.
        beta (float): Defaults to 1.0/9.0.
        eps (float): Defaults to 1e-6.

    Returns:
        loss (torch.Tensor)
    """
    pred = pred * torch.tensor([1, 1, 1, 1, 90], device=pred.device)
    pred = pred.to(torch.float64)
    target = target * torch.tensor([1, 1, 1, 1, 90], device=target.device)
    target = target.to(torch.float64)
    xy_p = pred[:, :2]
    xy_t = target[:, :2]
    _, Sigma_p = xy_wh_r_2_xy_sigma(pred)
    _, Sigma_t = xy_wh_r_2_xy_sigma(target)

    # Smooth-L1 norm
    diff = torch.abs(xy_p - xy_t)
    xy_loss = torch.where(diff < beta, 0.5 * diff * diff / beta,
                          diff - 0.5 * beta).sum(dim=-1)
    Vb_p = 4 * Sigma_p.det().sqrt()
    Vb_t = 4 * Sigma_t.det().sqrt()
    K = Sigma_p.bmm((Sigma_p + Sigma_t).inverse())
    Sigma = Sigma_p - K.bmm(Sigma_p)
    Vb = 4 * Sigma.det().sqrt()
    Vb = torch.where(torch.isnan(Vb), torch.full_like(Vb, 0), Vb)
    KFIoU = Vb / (Vb_p + Vb_t - Vb + eps)
    if fun == 'ln':
        kf_loss = -torch.log(KFIoU + eps)
    elif fun == 'exp':
        kf_loss = torch.exp(1 - KFIoU) - 1
    else:
        kf_loss = 1 - 3 * KFIoU

    loss = (xy_loss + kf_loss).clamp(0)

    return loss





def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio,dn="no",dn_losses=[], panoptic_on=False, semantic_ce_loss=False, duplicate_box_matching=False, live_cell=False, num_gt_points = 0, dn_aug = False, crop = False, simi=False):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        self.dn=dn
        self.dn_aug = dn_aug
        self.dn_losses=dn_losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.num_points = num_points
        self.num_gt_points = num_gt_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio
        self.focal_alpha = 0.25

        self.panoptic_on = panoptic_on
        self.semantic_ce_loss = semantic_ce_loss
        self.duplicate_box_matching = duplicate_box_matching
        self.live_cell = live_cell
        self.crop = crop
        self.simi = simi

    def loss_labels_ce(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(
            src_logits.shape[:2], self.num_classes, dtype=torch.int64, device=src_logits.device
        )
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {"loss_ce": loss_ce}
        return losses

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        try:
            target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        except:
            print(indices)
            print(targets)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)  #将第二类视为背景类
        target_classes[idx] = target_classes_o

        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses


    def loss_labels_duplicate(self, outputs, targets, indices, indices_duplicate, num_boxes, item = False, log=True):
        """Classification loss (Binary focal loss)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        idx_duplicate = self._get_src_permutation_idx(indices_duplicate)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes_1 = torch.zeros_like(indices_duplicate[0][1], device=target_classes_o.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        target_classes[idx_duplicate] = target_classes_1
        target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]+1],
                                            dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
        target_classes_onehot.scatter_(2, target_classes.unsqueeze(-1), 1)

        target_classes_onehot = target_classes_onehot[:,:,:-1]
        loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_boxes, alpha=self.focal_alpha, gamma=2) * src_logits.shape[1]
        losses = {'loss_ce': loss_ce}

        return losses



    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_oboxes(self, outputs, targets, indices, num_boxes): 
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        scale_angle = torch.tensor([1.0, 1.0, 1.0, 1.0, 90.0], device=target_boxes.device)
        angle_convert = torch.tensor([1.0, 1.0, 1.0, 1.0, math.pi / 2.0], device= target_boxes.device)
        target_boxes = target_boxes / scale_angle
        target_boxes_detach = target_boxes.detach() * angle_convert
        src_boxes_detach = src_boxes.detach() * angle_convert
        iou_weight = 1 - box_iou_rotated(src_boxes_detach.float(), target_boxes_detach.float(), aligned=True)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox * iou_weight[:, None]
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = kfiou_loss(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_oboxes_duplicate(self, outputs, targets, indices, indices_duplicate, num_boxes): 
        assert 'pred_boxes' in outputs 
        idx = self._get_src_permutation_idx(indices)
        idx_duplicate = self._get_src_permutation_idx(indices_duplicate)
        src_boxes = outputs['pred_boxes'][idx]
        src_boxes_duplicate = outputs['pred_boxes'][idx_duplicate]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_boxes_duplicate = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices_duplicate)], dim=0)
        src_boxes = torch.cat((src_boxes, src_boxes_duplicate), dim=0)
        target_boxes = torch.cat((target_boxes, target_boxes_duplicate), dim=0)
        scale_angle = torch.tensor([1.0, 1.0, 1.0, 1.0, 90.0], device=target_boxes.device)
        angle_convert = torch.tensor([1.0, 1.0, 1.0, 1.0, math.pi / 2.0], device= target_boxes.device)
        
        target_boxes = target_boxes / scale_angle
        target_boxes_detach = target_boxes.detach() * angle_convert
        src_boxes_detach = src_boxes.detach() * angle_convert
        
        iou = 1 - box_iou_rotated(src_boxes_detach.float(), target_boxes_detach.float())
        iou_weight = torch.diag(iou)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        loss_bbox = loss_bbox * iou_weight[:, None]
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = kfiou_loss(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_boxes_panoptic(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        target_labels = torch.cat([t['labels'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        isthing=target_labels<80
        target_boxes=target_boxes[isthing]
        src_boxes=src_boxes[isthing]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks[tgt_idx]
        target_masks = target_masks.to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_crop(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]

        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        mask_pos = [torch.tensor(t["mask_pos"]) for t in targets]
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks[tgt_idx]
        target_masks = target_masks.to(src_masks)

        max_length = max([tensor.size(0) for tensor in mask_pos])
        mask_pos = torch.cat([torch.nn.functional.pad(tensor, (0, 0, 0, max_length - tensor.size(0))).unsqueeze(0) for tensor in mask_pos], dim=0)
        target_mask_pos = mask_pos[tgt_idx].to(src_masks.device)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        src_masks = F.interpolate(
                src_masks,
                size=list(targets[0]["mask_size"]),
                mode="bilinear",
                align_corners=False,
            )
        src_masks = torch.stack([
                src_masks[i, 0, target_mask_pos[i, 2]:target_mask_pos[i, 3], target_mask_pos[i, 0]:target_mask_pos[i, 1]]
                for i in range(src_masks.shape[0])
            ])
        src_masks = src_masks[:, None]
        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_polygons(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        tgt_obboxes = targets[0]["boxes"].unsqueeze(0)
        # assume batch = 1
        tgt_obboxes = tgt_obboxes[tgt_idx]
        tgt_idx = list(tgt_idx)
        tgt_idx = torch.cat((tgt_idx[0].unsqueeze(-1), tgt_idx[1].unsqueeze(-1)), dim=1).tolist()
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        polygons = [t["masks"].polygons for t in targets]
        target_polygons = [polygons[i[0]][i[1]] for i in tgt_idx]

        target_masks = convert_coco_poly_to_mask(target_polygons, 1024, 1024)
        # TODO use valid to mask invalid areas due to padding in loss
        #target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords through pointrender
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # sample point_coords according to gt mask
            point_coords_tgt = get_point_coords_with_gt_mask(tgt_obboxes, self.num_gt_points).to(torch.float32)
            # get gt labels
            point_coords = torch.cat((point_coords, point_coords_tgt), dim=1)
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def loss_masks_duplicate(self, outputs, targets, indices, indices_duplicate, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_idx_duplicate = self._get_src_permutation_idx(indices_duplicate)
        tgt_idx_duplicate = self._get_tgt_permutation_idx(indices_duplicate)
        src_masks = outputs["pred_masks"]
        src_masks = torch.cat((src_masks[src_idx], src_masks[src_idx_duplicate]), dim=0)
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = torch.cat((target_masks[tgt_idx], target_masks[tgt_idx_duplicate]), dim=0)

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        with torch.no_grad():
            # sample point_coords
            point_coords = get_uncertain_point_coords_with_randomness(
                src_masks,
                lambda logits: calculate_uncertainty(logits),
                self.num_points,
                self.oversample_ratio,
                self.importance_sample_ratio,
            )
            # get gt labels
            point_labels = point_sample(
                target_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        point_logits = point_sample(
            src_masks,
            point_coords,
            align_corners=False,
        ).squeeze(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def prep_for_dn(self,mask_dict):
        output_known_lbs_bboxes = mask_dict['output_known_lbs_bboxes']

        known_indice = mask_dict['known_indice']
        scalar,pad_size=mask_dict['scalar'],mask_dict['pad_size']
        assert pad_size % scalar==0
        single_pad=pad_size//scalar

        num_tgt = known_indice.numel()
        return output_known_lbs_bboxes,num_tgt,single_pad,scalar

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss_duplicate(self, loss, outputs, targets, indices, indices_duplicate, indices_duplicate_labels, num_masks, num_labels, item=False):
        loss_map = {
            'labels': self.loss_labels_duplicate,
            'masks': self.loss_masks_duplicate,
            'boxes': self.loss_oboxes_duplicate,
            'oboxes': self.loss_oboxes_duplicate,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        if loss == 'labels':
            return loss_map[loss](outputs, targets, indices, indices_duplicate_labels, num_labels, item)
        return loss_map[loss](outputs, targets, indices, indices_duplicate, num_masks)


    def get_loss(self, loss, outputs, targets, indices, num_masks, second=False):
        loss_masks = self.loss_masks
        loss_label = self.loss_labels
        if self.live_cell:
            loss_masks = self.loss_masks_polygons
        elif self.crop:
            loss_masks = self.loss_masks_crop
        if self.semantic_ce_loss:
            loss_label = self.loss_labels_ce
        loss_map = {
            'labels': loss_label,
            'masks': loss_masks, 
            'boxes': self.loss_boxes_panoptic if self.panoptic_on else self.loss_boxes,
            'oboxes': self.loss_oboxes,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)

    def forward(self, outputs, targets, mask_dict=None, is_track = False, tracker = None):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
            is_track
            tracker
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs" and k!= "pred_motion_cls" and k != "pred_motion"}
        losses = {}
        # Retrieve the matching between the outputs of the last layer and the targets
        if self.dn != "no" and mask_dict != None:
            output_known_lbs_bboxes,num_tgt,single_pad,scalar = self.prep_for_dn(mask_dict)
            exc_idx = []
            for i in range(len(targets)):
                if len(targets[i]['labels']) > 0:
                    t = torch.arange(0, len(targets[i]['labels'])).long().cuda()
                    t = t.unsqueeze(0).repeat(scalar, 1)
                    tgt_idx = t.flatten()
                    if self.dn_aug:
                        num = len(targets[i]['labels'])
                        tgt_pad = num * 2
                        # num = int(single_pad / 2)
                        #output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                        dn_boxes = output_known_lbs_bboxes["pred_boxes"][i, :single_pad*scalar]
                        target_boxes = targets[i]["boxes"].repeat(scalar * 2, 1)
                        scale_angle = torch.tensor([1.0, 1.0, 1.0, 1.0, 90.0], device=target_boxes.device)
                        angle_convert = torch.tensor([1.0, 1.0, 1.0, 1.0, math.pi / 2.0], device= target_boxes.device)
                        target_boxes = target_boxes / scale_angle 
                        t = torch.arange(0, len(targets[i]['labels']) * 2).long().cuda()
                        t = t.unsqueeze(0).repeat(scalar, 1)
                        all_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                        all_idx = all_idx.flatten()
                        dn_boxes = dn_boxes
                        dn_boxes = dn_boxes[all_idx]
                        similarity = 1 - kfiou_loss(dn_boxes, target_boxes)
                        output_idx = torch.zeros(num * scalar, device=target_boxes.device)
                        for j in range(scalar):
                            value = similarity[tgt_pad * j : num + tgt_pad *j] - similarity[tgt_pad * j + num : tgt_pad + tgt_pad *j]
                            all_idx[tgt_pad * j : num + tgt_pad * j] = all_idx[tgt_pad * j : num + tgt_pad * j] * (value > 0)
                            all_idx[num + tgt_pad * j : tgt_pad + tgt_pad * j] = all_idx[num + tgt_pad * j : tgt_pad + tgt_pad * j] * (value < 0)  
                            output_idx[num * j : num * (j+1)] = all_idx[tgt_pad * j : num + tgt_pad * j] + all_idx[num + tgt_pad * j : tgt_pad + tgt_pad * j]
                    else:    
                        output_idx = (torch.tensor(range(scalar)) * single_pad).long().cuda().unsqueeze(1) + t
                        output_idx = output_idx.flatten()
                    
                    if is_track:
                        num = len(targets[i]['labels'])
                else:
                    output_idx = tgt_idx = torch.tensor([]).long().cuda()
                exc_idx.append((output_idx.long(), tgt_idx))
        if self.live_cell:
            cost = [["cls", self.losses[2]]]
        else:
            cost = ["cls", self.losses[2], "mask"]
        second = False
        reverse = False
        pseudo = False
        if self.simi:
            reverse = tracker.reverse
            pseudo = tracker.pseudo
        if is_track and tracker.track_num > 0:  
            second = True
            num_obj_query = targets[0]["num_query"]
            num_max_track_query = (outputs_without_aux["pred_boxes"].shape[1] - num_obj_query)
            object_outputs = {}
            track_outputs = {}
            for i in outputs_without_aux:
                if i != 'interm_outputs' and i!="num_select" and i!= "counting_output" and i != "dq_outputs" and i!= 'track_outputs':
                    object_outputs[i] = outputs_without_aux[i][:, :num_obj_query]
                    track_outputs[i] = outputs_without_aux[i][:, num_obj_query:]
            object_targets = []
            track_output_indices = []
            target_output_indices = []
            object_target_indices = []
            track_output_indices_only = []
            track_motion_output_indices = []
            motion_cls_targets = []
            simi_track_indices = []
            for i in range(len(targets)):
                object_target = {}
                track_ids = targets[i]["track_id"]
                target_ids = targets[i]["gt_id"]
                num_target = len(target_ids)
                motion_cls_target = torch.zeros_like(track_ids, device=track_ids.device)
                combined = torch.cat((track_ids,target_ids), dim=0)
                combined_val, counts = combined.unique(return_counts=True)
                common_elements = combined_val[counts>1]
                num_valid = len(common_elements)
                indices_track = torch.tensor([torch.where(track_ids == x)[0][0] for x in common_elements], device=track_ids.device, dtype=torch.int64)
                simi_track_indice = indices_track.clone()  
                indices_target = torch.tensor([torch.where(target_ids == x)[0][0] for x in common_elements], device=target_ids.device, dtype=torch.int64)
                all_target_indices = torch.arange(num_target, device=indices_target.device, dtype=torch.int64)
                object_target_indice = torch.tensor([x not in indices_target for x in all_target_indices])
                object_target_indice = all_target_indices[object_target_indice==1]
                track_output_indice = indices_track
                target_output_indice = indices_target[:num_valid]
                for j in ["labels", "masks", "boxes"]:
                    object_target[j] = targets[i][j][object_target_indice]
                object_targets.append(object_target)    
                track_motion_output_indices.append((track_output_indice, target_output_indice))
                track_output_indices.append(track_output_indice + num_obj_query)
                track_output_indices_only.append(track_output_indice)
                target_output_indices.append(target_output_indice.to(torch.int64))
                object_target_indices.append(object_target_indice.to(torch.int64))
                motion_cls_targets.append(motion_cls_target.unsqueeze(-1))
                simi_track_indices.append(simi_track_indice)
            if not reverse and self.simi:
                cost = ["mask_simi"]
            indices, cost_matrix = self.matcher(object_outputs, object_targets, cost = cost)
            indices_track_only = []
            for index in range(len(indices)):
                track_indices = torch.cat((indices[index][0].to(track_output_indices[index].device), track_output_indices[index]) , dim=0)
                target_indices = torch.cat((object_target_indices[index][indices[index][1]].to(target_output_indices[index].device), target_output_indices[index]), dim=0)
                indices[index] = (track_indices.to(torch.int64), target_indices.to(torch.int64))
                indices_track_only.append((track_output_indices_only[index], target_output_indices[index]))
            del track_outputs
            del object_outputs
        else:
            indices, cost_matrix = self.matcher(outputs_without_aux, targets, cost=cost)         
            if is_track:  
                track_ids = []
                track_moists = []
                track_pos = []
                track_query = []
                max_num = tracker.max_num
                for i in range(len(targets)):
                    src_index = indices[i][0]
                    tgt_index = indices[i][1]
                    track_ids.append(targets[i]["gt_id"][tgt_index])
                    track_moists.append(targets[i]["moist"])
                    item_pos = outputs_without_aux["pred_boxes"][i][src_index].detach()
                    padding_pos = torch.zeros((max_num - item_pos.shape[0], item_pos.shape[1]), device = outputs_without_aux["pred_boxes"].device)
                    item_pos = torch.cat((item_pos, padding_pos), dim=0)
                    item_query = outputs_without_aux["query"][i][src_index].detach()
                    padding_query = torch.zeros((max_num - item_query.shape[0], item_query.shape[1]), device = outputs_without_aux["query"].device)
                    item_query = torch.cat((item_query, padding_query), dim=0)
                    track_pos.append(item_pos.unsqueeze(0))
                    track_query.append(item_query.unsqueeze(0))
                tracker.track_pos = torch.cat(track_pos, dim=0)
                tracker.track_query = torch.cat(track_query, dim=0)
                tracker.moists = track_moists
                tracker.track_ids = track_ids
                tracker.track_num = len(track_ids)
                del track_query
        num_masks = min(sum(len(t["labels"]) for t in targets), outputs_without_aux["pred_boxes"].shape[1]*outputs_without_aux["pred_boxes"].shape[0])  #当query数量较少时
        num_masks = torch.as_tensor(
            [num_masks], dtype=torch.float, device=next(iter(outputs.values())).device
        )
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_masks)
        num_masks = torch.clamp(num_masks / get_world_size(), min=1).item()
        if self.duplicate_box_matching:
            bs, num_queries = outputs_without_aux["pred_logits"].shape[:2]
            mask_indices = torch.ones((bs, num_queries), dtype=torch.bool)
            indices_tmp = torch.tensor([indice[0].tolist() for indice in indices])
            mask_indices.scatter_(1, indices_tmp, False)
            last_indices = [torch.nonzero(mask_indice).squeeze(dim=1) for mask_indice in mask_indices]
            last_iou =  torch.cat([giou[last_indice].unsqueeze(dim=0) for giou, last_indice in zip(cost_matrix, last_indices)], dim=0).to("cpu")
            mask_k_values, mask_k_indices = torch.topk(last_iou, 3, dim=1)
            last_zero_iou = torch.zeros_like(last_iou)
            last_iou_0 = last_zero_iou.scatter_(1, mask_k_indices, mask_k_values)
            max_values, max_indices = torch.max(last_iou_0, dim=2)
            max_values_all, max_indices_all = torch.max(last_iou, dim=2)
            mask_valid_indices = [torch.nonzero(max_value > 0).squeeze(dim=1) for max_value in max_values]
            mask_valid_label_indices = [torch.nonzero(max_value > 0).squeeze(dim=1) for max_value in max_values_all]
            indices_duplicate_label = [(last_indices[i][mask_valid_indice], max_indices_all[i][mask_valid_indice]) for i,mask_valid_indice in enumerate(mask_valid_label_indices)]
            indices_duplicate = [(last_indices[i][mask_valid_indice], max_indices[i][mask_valid_indice]) for i,mask_valid_indice in enumerate(mask_valid_indices)]
            num_masks_duplicate = num_masks + sum(indice_duplicate[1].shape[0] for indice_duplicate in indices_duplicate)
            num_labels = num_masks + sum(indice_duplicate[1].shape[0] for indice_duplicate in indices_duplicate_label)
        # Compute all the requested losses
        if pseudo:
            pseudo_targets = []
            for i in range(len(targets)):
                print(len(indices[0][1]))
                pseudo_target = {}
                src_idx = self._get_src_permutation_idx(indices)
                src_masks = outputs["pred_masks"][src_idx]
                src_masks = F.interpolate(
                    src_masks.unsqueeze(0),
                    size=(targets[0]["masks"].shape[-2], targets[0]["masks"].shape[-1]),
                    mode="bilinear",
                    align_corners=False,
                    ).squeeze(0)
                src_boxes = outputs['pred_boxes'][src_idx]
                n = src_masks.shape[0]
                scale_angle = torch.tensor([1.0, 1.0, 1.0, 1.0, 90.0], device=src_boxes.device)
                src_boxes = src_boxes * scale_angle
                src_labels = torch.zeros(n, dtype=torch.int64,device=src_masks.device)
                tgt_index = indices[i][1]
                src_ids = targets[i]["gt_id"][tgt_index]
                src_masks[src_masks>0] = 1
                src_masks[src_masks<0] = 0
                pseudo_target["masks"] = src_masks.clone().detach()
                pseudo_target["boxes"] = src_boxes.clone().detach()
                pseudo_target["labels"] = src_labels.clone().detach()
                pseudo_target["gt_ids"] = src_ids.clone().detach()
                pseudo_targets.append(pseudo_target)
            return {}, [pseudo_target]

        del outputs_without_aux
        for loss in self.losses:
            if self.duplicate_box_matching:
                losses.update(self.get_loss_duplicate(loss, outputs, targets, indices, indices_duplicate, indices_duplicate_label, num_masks_duplicate, num_labels))
            else:
                losses.update(self.get_loss(loss, outputs, targets, indices, num_masks, second=second))
    
        if self.dn != "no" and mask_dict != None:
            l_dict={}
            for loss in self.dn_losses:
                # num_targets = len(targets)
                # cost_matrix = cost_matrix_compute(output_known_lbs_bboxes, targets, cost = [["cls", self.losses[2]]])
                l_dict.update(self.get_loss(loss, output_known_lbs_bboxes, targets, exc_idx, num_masks*scalar))
            l_dict = {k + f'_dn': v for k, v in l_dict.items()}
            losses.update(l_dict)
        elif self.dn != "no":
            l_dict = dict()
            l_dict['loss_bbox_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_giou_dn'] = torch.as_tensor(0.).to('cuda')
            l_dict['loss_ce_dn'] = torch.as_tensor(0.).to('cuda')
            if self.dn == "seg":
                l_dict['loss_mask_dn'] = torch.as_tensor(0.).to('cuda')
                l_dict['loss_dice_dn'] = torch.as_tensor(0.).to('cuda')
            losses.update(l_dict)

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if is_track and len(targets[0]["track_id"]) > 0:
                    object_outputs = {}
                    for j in aux_outputs:
                        object_outputs[j] = aux_outputs[j][:, :num_obj_query]
                    if i==0:
                        indices, cost_matrix = self.matcher(object_outputs, targets, cost = cost)
                    else:
                        indices, cost_matrix = self.matcher(object_outputs, object_targets, cost = cost)
                        del object_outputs
                        for index in range(len(indices)):
                            track_indices = torch.cat((indices[index][0].to(track_output_indices[index].device), track_output_indices[index]) , dim=0)
                            target_indices = torch.cat((object_target_indices[index][indices[index][1]].to(track_output_indices[index].device), target_output_indices[index]), dim=0)
                            # if self.simi:
                            #     indices[index] = (track_indices, target_indices, simi_track_indices[index], target_output_indices[index])  #加入前后两帧都有的
                            # else:
                            indices[index] = (track_indices, target_indices)                
                else:
                    indices, cost_matrix = self.matcher(aux_outputs, targets, cost = cost)
                
                #indices, cost_matrix = self.matcher(aux_outputs, targets, cost = ["cls", self.losses[2], "mask"])
                if self.duplicate_box_matching:
                    bs, num_queries = aux_outputs["pred_logits"].shape[:2]
                    mask_indices = torch.ones((bs, num_queries), dtype=torch.bool)
                    indices_tmp = torch.tensor([indice[0].tolist() for indice in indices])
                    mask_indices.scatter_(1, indices_tmp, False)
                    last_indices = [torch.nonzero(mask_indice).squeeze(dim=1) for mask_indice in mask_indices]
                    last_iou =  torch.cat([giou[last_indice].unsqueeze(dim=0) for giou, last_indice in zip(cost_matrix, last_indices)], dim=0).to("cpu")
                    mask_k_values, mask_k_indices = torch.topk(last_iou, 3, dim=1)
                    last_zero_iou = torch.zeros_like(last_iou)
                    last_iou_0 = last_zero_iou.scatter_(1, mask_k_indices, mask_k_values)
                    max_values, max_indices = torch.max(last_iou_0, dim=2)
                    max_values_all, max_indices_all = torch.max(last_iou, dim=2)
                    mask_valid_indices = [torch.nonzero(max_value > 0).squeeze(dim=1) for max_value in max_values]
                    mask_valid_label_indices = [torch.nonzero(max_value > 0).squeeze(dim=1) for max_value in max_values_all]
                    indices_duplicate_label = [(last_indices[i][mask_valid_indice], max_indices_all[i][mask_valid_indice]) for i,mask_valid_indice in enumerate(mask_valid_label_indices)]
                    indices_duplicate = [(last_indices[i][mask_valid_indice], max_indices[i][mask_valid_indice]) for i,mask_valid_indice in enumerate(mask_valid_indices)]
                    num_masks_duplicate = num_masks + sum(indice_duplicate[1].shape[0] for indice_duplicate in indices_duplicate)
                    num_labels = num_masks + sum(indice_duplicate[1].shape[0] for indice_duplicate in indices_duplicate_label)
                for loss in self.losses:
                    if self.duplicate_box_matching:
                        l_dict =self.get_loss_duplicate(loss, aux_outputs, targets, indices, indices_duplicate, indices_duplicate_label, num_masks_duplicate, num_labels)
                    else:
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks, second=second)
                    l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                    losses.update(l_dict)
                if 'interm_outputs' in outputs:
                    start = 0
                else:
                    start = 1
                if i>=start:
                    if self.dn != "no" and mask_dict is not None:
                        out_=output_known_lbs_bboxes['aux_outputs'][i]
                        l_dict = {}
                        for loss in self.dn_losses:
                            l_dict.update(
                                self.get_loss(loss, out_, targets, exc_idx, num_masks * scalar))
                        l_dict = {k + f'_dn_{i}': v for k, v in l_dict.items()}
                        losses.update(l_dict)
                    elif self.dn != "no":
                        l_dict = dict()
                        l_dict[f'loss_bbox_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_giou_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        l_dict[f'loss_ce_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        if self.dn == "seg":
                            l_dict[f'loss_mask_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                            l_dict[f'loss_dice_dn_{i}'] = torch.as_tensor(0.).to('cuda')
                        losses.update(l_dict)
        # interm_outputs loss
        if 'interm_outputs' in outputs:
            interm_outputs = outputs['interm_outputs']
            #indices, cost_matrix = self.matcher(interm_outputs, targets, cost = ["cls", self.losses[2], "mask"])
            indices, cost_matrix = self.matcher(interm_outputs, targets, cost = cost)
            if self.duplicate_box_matching:
                bs, num_queries = aux_outputs["pred_logits"].shape[:2]
                mask_indices = torch.ones((bs, num_queries), dtype=torch.bool)
                indices_tmp = torch.tensor([indice[0].tolist() for indice in indices])
                mask_indices.scatter_(1, indices_tmp, False)
                last_indices = [torch.nonzero(mask_indice).squeeze(dim=1) for mask_indice in mask_indices]
                last_iou =  torch.cat([giou[last_indice].unsqueeze(dim=0) for giou, last_indice in zip(cost_matrix, last_indices)], dim=0).to("cpu")
                mask_k_values, mask_k_indices = torch.topk(last_iou, 3, dim=1)
                last_zero_iou = torch.zeros_like(last_iou)
                last_iou_0 = last_zero_iou.scatter_(1, mask_k_indices, mask_k_values)
                max_values, max_indices = torch.max(last_iou_0, dim=2)
                max_values_all, max_indices_all = torch.max(last_iou, dim=2)
                mask_valid_indices = [torch.nonzero(max_value > 0).squeeze(dim=1) for max_value in max_values]
                mask_valid_label_indices = [torch.nonzero(max_value > 0).squeeze(dim=1) for max_value in max_values_all]
                indices_duplicate_label = [(last_indices[i][mask_valid_indice], max_indices_all[i][mask_valid_indice]) for i,mask_valid_indice in enumerate(mask_valid_label_indices)]
                indices_duplicate = [(last_indices[i][mask_valid_indice], max_indices[i][mask_valid_indice]) for i,mask_valid_indice in enumerate(mask_valid_indices)]
                num_masks_duplicate = num_masks + sum(indice_duplicate[1].shape[0] for indice_duplicate in indices_duplicate)
                num_labels = num_masks + sum(indice_duplicate[1].shape[0] for indice_duplicate in indices_duplicate_label)
            if self.simi and second and not reverse:
                losses_dict = ["labels", "masks"]
            else:
                losses_dict = self.losses
            for loss in losses_dict:
                if self.duplicate_box_matching:
                    l_dict =self.get_loss_duplicate(loss, aux_outputs, targets, indices, indices_duplicate, indices_duplicate_label, num_masks_duplicate, num_labels, item=True)
                else:
                    l_dict = self.get_loss(loss, interm_outputs, targets, indices, num_masks, second=second)
                l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                losses.update(l_dict)

        if 'track_outputs' in outputs:
            track_outputs = outputs['track_outputs']
            num_masks = min(len(indices_track_only[0][0]), len(indices_track_only[0][1]))
            if num_masks>0:
                for loss in losses_dict:
                    l_dict = self.get_loss(loss, track_outputs, targets, indices_track_only, num_masks)
                    l_dict = {k + f'_track': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        if 'dq_outputs' in outputs and object_targets[0]["labels"].shape[0]>0:
            dq_outputs = outputs['dq_outputs']
            indices, cost_matrix = self.matcher(dq_outputs, object_targets, cost = cost)
            dq_loss = self.get_loss('labels', dq_outputs, object_targets, indices, num_masks=indices[0][1].shape[0])
            dq_loss = {k + f'_dq': v for k, v in dq_loss.items()}
            losses.update(dq_loss)
            if ~torch.isfinite(dq_loss['loss_ce_dq']).all():
                print("张量中包含 NaN 或 Inf")
        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher.__repr__(_repr_indent=8)),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
