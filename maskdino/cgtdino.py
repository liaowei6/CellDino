# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
from typing import Tuple

import torch
import math
from torch import nn
from torch.nn import functional as F
from collections import deque
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.modeling import META_ARCH_REGISTRY, build_backbone, build_sem_seg_head
from detectron2.modeling.backbone import Backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, RotatedBoxes
from detectron2.utils.memory import retry_if_cuda_oom

from .modeling.criterion import SetCriterion, kfiou_loss
from .modeling.matcher import HungarianMatcher
from .utils import box_ops
from .utils.utils import EBBoxes, BitMasks_ct
from mmcv.ops import box_iou_rotated

@META_ARCH_REGISTRY.register()
class CgtDINO(nn.Module):
    """
    Main class for mask classification semantic segmentation architectures.
    """

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        sem_seg_head: nn.Module,
        criterion: nn.Module,
        num_queries: int,
        object_mask_threshold: float,
        overlap_threshold: float,
        metadata,
        size_divisibility: int,
        sem_seg_postprocess_before_inference: bool,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        # inference
        semantic_on: bool,
        panoptic_on: bool,
        instance_on: bool,
        test_topk_per_image: int,
        data_loader: str,
        pano_temp: float,
        focus_on_box: bool = False,
        transform_eval: bool = False,
        semantic_ce_loss: bool = False,
        duplicate: bool = False,
        crop: bool = False,
        simi: bool = False,
        dq: bool = False,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            sem_seg_head: a module that predicts semantic segmentation from backbone features
            criterion: a module that defines the loss
            num_queries: int, number of queries
            object_mask_threshold: float, threshold to filter query based on classification score
                for panoptic segmentation inference
            overlap_threshold: overlap threshold used in general inference for panoptic segmentation
            metadata: dataset meta, get `thing` and `stuff` category names for panoptic
                segmentation inference
            size_divisibility: Some backbones require the input height and width to be divisible by a
                specific integer. We can use this to override such requirement.
            sem_seg_postprocess_before_inference: whether to resize the prediction back
                to original input size before semantic segmentation inference or after.
                For high-resolution dataset like Mapillary, resizing predictions before
                inference will cause OOM error.
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            semantic_on: bool, whether to output semantic segmentation prediction
            instance_on: bool, whether to output instance segmentation prediction
            panoptic_on: bool, whether to output panoptic segmentation prediction
            test_topk_per_image: int, instance segmentation parameter, keep topk instances per image
            transform_eval: transform sigmoid score into softmax score to make score sharper
            semantic_ce_loss: whether use cross-entroy loss in classification
        """
        super().__init__()
        self.backbone = backbone
        self.pano_temp = pano_temp
        self.sem_seg_head = sem_seg_head
        #冻结部分权重只训练预测头和相关层权重
        # 冻结所有层的权重
        # for param in self.sem_seg_head.parameters():
        #     param.requires_grad = False
        # self.sem_seg_head.predictor.class_embed.weighs.requires_grad = True
        # self.sem_seg_head.predictor.class_embed.bias.requires_grad = True
        # self.sem_seg_head.predictor.label_enc.weights.requires_grad = True
        # self.sem_seg_head.predictor.label_enc.bias.requires_grad = True

        self.criterion = criterion
        self.num_queries = num_queries
        self.overlap_threshold = overlap_threshold
        self.object_mask_threshold = object_mask_threshold
        self.metadata = metadata
        if size_divisibility < 0:
            # use backbone size_divisibility if not set
            size_divisibility = self.backbone.size_divisibility
        self.size_divisibility = size_divisibility
        self.sem_seg_postprocess_before_inference = sem_seg_postprocess_before_inference
        self.register_buffer("pixel_mean", torch.Tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.Tensor(pixel_std).view(-1, 1, 1), False)

        # additional args
        self.semantic_on = semantic_on
        self.instance_on = instance_on
        self.panoptic_on = panoptic_on
        self.test_topk_per_image = test_topk_per_image

        self.data_loader = data_loader
        self.focus_on_box = focus_on_box
        self.transform_eval = transform_eval
        self.semantic_ce_loss = semantic_ce_loss
        self.duplicate = duplicate
        self.crop = crop
        self.simi = simi
        self.dq = dq
        if not self.semantic_on:
            assert self.sem_seg_postprocess_before_inference

        print('criterion.weight_dict ', self.criterion.weight_dict)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        sem_seg_head = build_sem_seg_head(cfg, backbone.output_shape())

        # Loss parameters:
        deep_supervision = cfg.MODEL.MaskDINO.DEEP_SUPERVISION
        no_object_weight = cfg.MODEL.MaskDINO.NO_OBJECT_WEIGHT

        # loss weights
        class_weight = cfg.MODEL.MaskDINO.CLASS_WEIGHT
        cost_class_weight = cfg.MODEL.MaskDINO.COST_CLASS_WEIGHT
        cost_dice_weight = cfg.MODEL.MaskDINO.COST_DICE_WEIGHT
        dice_weight = cfg.MODEL.MaskDINO.DICE_WEIGHT  #
        cost_mask_weight = cfg.MODEL.MaskDINO.COST_MASK_WEIGHT  #
        mask_weight = cfg.MODEL.MaskDINO.MASK_WEIGHT
        cost_box_weight = cfg.MODEL.MaskDINO.COST_BOX_WEIGHT
        box_weight = cfg.MODEL.MaskDINO.BOX_WEIGHT  #
        cost_giou_weight = cfg.MODEL.MaskDINO.COST_GIOU_WEIGHT
        giou_weight = cfg.MODEL.MaskDINO.GIOU_WEIGHT  #
        motion_weight = cfg.MODEL.MaskDINO.MOTION_WEIGHT
        motion_cls_weight = cfg.MODEL.MaskDINO.MOTION_CLS_WEIGHT
        # img_lst_weight = cfg.MODEL.MaskDINO.IMG_LST_WEIGHT
        # fea_lst_weight = cfg.MODEL.MaskDINO.FEA_LST_WEIGHT
        # lcm_weight = cfg.MODEL.MaskDINO.LCM_WEIGHT
        # deep_lst_weight = cfg.MODEL.MaskDINO.DEEP_LST_WEIGHT
        # area_weight = cfg.MODEL.MaskDINO.AREA_WEIGHT
        # energy_weight = cfg.MODEL.MaskDINO.ENERGY_WEIGHT
        # building matcher
        matcher = HungarianMatcher(
            cost_class=cost_class_weight,
            cost_mask=cost_mask_weight,
            cost_dice=cost_dice_weight,
            cost_box=cost_box_weight,
            cost_giou=cost_giou_weight,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
        )

        weight_dict = {"loss_ce": class_weight}
        weight_dict.update({"loss_mask": mask_weight, "loss_dice": dice_weight})
        weight_dict.update({"loss_bbox":box_weight,"loss_giou":giou_weight})
        weight_dict.update({"loss_motion":motion_weight, "loss_motion_cls":motion_cls_weight})
        weight_dict.update({'loss_ce_dq': 1})
       # weight_dict.update({"loss_img_lst": img_lst_weight, "loss_fea_lst": fea_lst_weight, "loss_lcm": lcm_weight, "loss_deep_lst": deep_lst_weight, "loss_area": area_weight, "loss_energy": energy_weight})
        # two stage is the query selection scheme
        if cfg.MODEL.MaskDINO.TWO_STAGE:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)
        # denoising training
        dn = cfg.MODEL.MaskDINO.DN
        if dn == "standard":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items() if k!="loss_mask" and k!="loss_dice" })
            dn_losses=["labels","boxes"]
        elif dn == "seg":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks","boxes"]
        elif dn == "ct":
            weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})
            dn_losses=["labels", "masks","oboxes"]
        else:           
            dn_losses=[]
        if cfg.TRACK:
            track_weight_dict = {}
            track_weight_dict.update({k + f'_track': v for k, v in weight_dict.items()})
            weight_dict.update(track_weight_dict)
        if deep_supervision:
            dec_layers = cfg.MODEL.MaskDINO.DEC_LAYERS
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        if cfg.MODEL.MaskDINO.BOX_LOSS:
            losses = ["labels", "masks","boxes"]
        elif cfg.MODEL.MaskDINO.OBOX_LOSS:
            losses = ["labels", "masks","oboxes"]
        else:
            losses = ["labels", "masks"]
        # building criterion
        criterion = SetCriterion(
            sem_seg_head.num_classes,
            matcher=matcher,
            weight_dict=weight_dict,
            eos_coef=no_object_weight,
            losses=losses,
            num_points=cfg.MODEL.MaskDINO.TRAIN_NUM_POINTS,
            oversample_ratio=cfg.MODEL.MaskDINO.OVERSAMPLE_RATIO,
            importance_sample_ratio=cfg.MODEL.MaskDINO.IMPORTANCE_SAMPLE_RATIO,
            dn=cfg.MODEL.MaskDINO.DN,
            dn_aug = cfg.MODEL.MaskDINO.DN_AUG,
            dn_losses=dn_losses,
            panoptic_on=cfg.MODEL.MaskDINO.PANO_BOX_LOSS,
            semantic_ce_loss=cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            #采用重复框匹配
            duplicate_box_matching = cfg.MODEL.MaskDINO.DUPLICATE_BOX_MATCHING,
            live_cell= cfg.INPUT.LIVE_CELL,
            num_gt_points = cfg.MODEL.MaskDINO.TRAIN_GT_POINTS,
            crop = cfg.CROP,
            simi = cfg.SIMI,
        )
        
        return {
            "backbone": backbone,
            "sem_seg_head": sem_seg_head,
            "criterion": criterion,
            "num_queries": cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES,
            "object_mask_threshold": cfg.MODEL.MaskDINO.TEST.OBJECT_MASK_THRESHOLD,
            "overlap_threshold": cfg.MODEL.MaskDINO.TEST.OVERLAP_THRESHOLD,
            "metadata": MetadataCatalog.get(cfg.DATASETS.TRAIN[0]),
            "size_divisibility": cfg.MODEL.MaskDINO.SIZE_DIVISIBILITY,
            "sem_seg_postprocess_before_inference": (
                cfg.MODEL.MaskDINO.TEST.SEM_SEG_POSTPROCESSING_BEFORE_INFERENCE
                or cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
                or cfg.MODEL.MaskDINO.TEST.INSTANCE_ON
            ),
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            # inference
            "semantic_on": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON,
            "instance_on": cfg.MODEL.MaskDINO.TEST.INSTANCE_ON,
            "panoptic_on": cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "data_loader": cfg.INPUT.DATASET_MAPPER_NAME,
            "focus_on_box": cfg.MODEL.MaskDINO.TEST.TEST_FOUCUS_ON_BOX,
            "transform_eval": cfg.MODEL.MaskDINO.TEST.PANO_TRANSFORM_EVAL,
            "pano_temp": cfg.MODEL.MaskDINO.TEST.PANO_TEMPERATURE,
            "semantic_ce_loss": cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and not cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON,
            "duplicate": cfg.MODEL.MaskDINO.DUPLICATE_BOX_MATCHING,
            "crop": cfg.CROP,
            "simi": cfg.SIMI,
            "dq": cfg.DQ,
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, tracker = None):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper`. 
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:
                   * "image": Tensor, image in (C, 2, H, W) format. 连续两帧图像
                   * "instances": per-region ground truth
                   * Other information that's included in the original dicts, such as:
                     "height", "width" (int): the output resolution of the model (may be different
                     from input resolution), used in inference.
            tracker: 细胞追踪轨迹
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:

                * "sem_seg":
                    A Tensor that represents the
                    per-pixel segmentation prediced by the head.
                    The prediction has shape KxHxW that represents the logits of
                    each class for each pixel.
                * "panoptic_seg":
                    A tuple that represent panoptic output
                    panoptic_seg (Tensor): of shape (height, width) where the values are ids for each segment.
                    segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                        Each dict contains keys "id", "category_id", "isthing".
        """
        images = [x["image"].to(self.device) for x in batched_inputs]  #batch
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.size_divisibility)
        padding_mask = [x["padding_mask"].to(self.device) for x in batched_inputs]

        #冻结encoder参数，只对Deoder进行训练
        with torch.no_grad():
            features = self.backbone(images.tensor)

        if self.training:
            # dn_args={"scalar":30,"noise_scale":0.4}
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                moists = [x["moists"] for x in batched_inputs]
                #计算动态查询数量
                if self.dq:
                    ccm_targets = []
                    ccm_params = [100, 300, 500]
                    for i in range(len(gt_instances)):
                        tgt_num = gt_instances[i].gt_classes.shape[0]
                        t = 0
                        for j in range(len(ccm_params)):
                            if tgt_num >= ccm_params[j]:
                                t = j + 1
                    ccm_targets.append(t) 
                    ccm_targets = torch.tensor(ccm_targets, dtype=torch.int64).to("cuda")
                    self.num_queries = ccm_params[torch.max(ccm_targets).item()]
                if "live_cell" in self.data_loader:
                     targets = self.prepare_targets_live_cell(gt_instances, images)
                elif 'detr' in self.data_loader:
                    if self.crop:
                        targets = self.prepare_targets_crop(gt_instances, images)
                    elif self.simi and not tracker.reverse:
                        # norm_images = [x["norm_image"].to(self.device) for x in batched_inputs]
                        # targets = self.prepare_targets_simi(gt_instances, images, moists, tracker, norm_images) 对于采用水平集损失
                        targets = self.prepare_targets_detr(gt_instances, images, moists, tracker)
                    else:
                        targets = self.prepare_targets_detr(gt_instances, images, moists, tracker) #对边框进行归一化，角度没有进行归一化
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            outputs, mask_dict = self.sem_seg_head(features, targets=targets, tracker=tracker)
            if tracker.pseudo:
                losses, pseudo_targets = self.criterion(outputs, targets, mask_dict, is_track=True, tracker=tracker)
                return pseudo_targets
            else:   
                losses = self.criterion(outputs, targets, mask_dict, is_track=True, tracker=tracker)
                for k in list(losses.keys()):
                    if k in self.criterion.weight_dict:
                        losses[k] *= self.criterion.weight_dict[k]
                    else:
                     # remove this loss if not specified in `weight_dict`
                        losses.pop(k)
                #ccm loss
            if self.dq:
                CCM_LOSS = torch.nn.CrossEntropyLoss()
                loss_ccm = CCM_LOSS(outputs["counting_output"], ccm_targets)
                losses['loss_ccm'] = loss_ccm * 1.0 
                # with open("num_acc.txt", "a", encoding="utf-8") as file:
                #     file.write(f"{outputs['num_select']} {ccm_params[ccm_targets]}\n")  # 写入 "a b" 并换行
            return losses
        else:
            outputs, _ = self.sem_seg_head(features, tracker=tracker)
            if self.dq:
                self.num_queries = outputs["num_select"]
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            mask_query = outputs["query"]
            # test
            #item_output = outputs["interm_outputs"]
            track_id = torch.tensor([], dtype=torch.int64)
            tracked_index = []
            min_center_distance = []
            # 对追踪query进行处理
            # print("-----" + str(tracker.frame_index) + "------")
            # print(tracker.track_ids)
            if tracker.track_num > 0:
                #超参数设置
                moist_min_area = tracker.moist_min_area
                track_min_area = tracker.track_min_area
                min_distance = tracker.moist_min_center_distance
                max_distance = tracker.moist_max_center_distance
                t_o_thresh = tracker.track_obj_score_thresh
                track_num = tracker.track_num
                track_cls_results = mask_cls_results[:, self.num_queries:, :].sigmoid()
                track_box_results = mask_box_results[:, self.num_queries:, :]
                track_pred_results = mask_pred_results[:, self.num_queries:, :, :]
                index = []
                track_id = []
                track_query = mask_query[:, self.num_queries:, :]
                angle_convert = torch.tensor([1.0, 1.0, 1.0, 1.0, math.pi / 2.0], device= track_box_results.device)
                first_boxes_results= track_box_results[0, :track_num, :] * angle_convert
                second_boxes_results = track_box_results[0, track_num:, :] * angle_convert
                iou = box_iou_rotated(first_boxes_results.float(), second_boxes_results.float(), aligned=True)  #无法做到完全精确,即对于某些旋转边框,即是完全相同也计算出也为0
                #采用中心点之间的距离来辅助判断
                # 计算欧氏距离  
                diff = first_boxes_results[:, :2] - second_boxes_results[:, :2]  # 计算差值
                center_distance = torch.sqrt(torch.sum(diff ** 2, dim=-1))  # 计算欧氏距离 
                max_index = track_cls_results[:, :track_num, 0] - track_cls_results[:, track_num:, 0]
                max_index = max_index.squeeze(0)
                max_index[max_index > 0] = 0
                max_index[max_index < 0] = 1
                track_index = torch.arange(0, track_num).to(max_index.device)
                track_index = track_index + max_index * track_num
                track_index = track_index.long()
                # angle_convert = torch.tensor([1.0, 1.0, 1.0, 1.0, math.pi / 2.0], device= mask_box_results.device)
                for i in range(track_num):
                    #if tracker.dataset=="DIC-C2DH-HeLa" and tracker.dataset_index=="01_RES" and tracker.track_ids[i]==20 and tracker.frame_index==75:
                    if tracker.dataset=="DIC-C2DH-HeLa" and tracker.dataset_index=="01_RES" and ((tracker.track_ids[i]==23 and tracker.frame_index==75) or (tracker.track_ids[i]==2 and tracker.frame_index==111)):
                        tracker.track_to_lost(tracker.tracks[i] ,tracker.frame_index)
                        continue
                    if track_cls_results[0, i, 0] > t_o_thresh and track_cls_results[0, i + track_num, 0] > t_o_thresh  and track_box_results[0, i, 2] * track_box_results[0][i][3] > moist_min_area and track_box_results[0, i+track_num, 2] * track_box_results[0][i+track_num][3] > moist_min_area:
                        #未发生分裂
                        if iou[i] > tracker.track_nms_thresh or center_distance[i] < min_distance or center_distance[i] > max_distance:
                            if tracker.tracks[i].state==0:
                                index.append(i)
                                tracker.track_to_active(tracker.tracks[i], track_query[0, i, :], track_cls_results[0, i, :], track_box_results[0, i, :])
                                track_id.append(tracker.tracks[-1].id)
                                tracked_index.append(len(tracker.tracks) -1)
                            else:
                                index.append(track_index[i])
                                track_id.append(tracker.tracks[i].id)
                                tracked_index.append(i)                            
                        #发生分裂
                        else:
                            tracker.track_to_lost(tracker.tracks[i],tracker.frame_index)
                            tracker.add_track(track_box_results[0, i, :], track_cls_results[0, i, :] ,track_query[0, i, :], tracker.tracks[i].id)
                            tracker.add_track(track_box_results[0, i+track_num, :], track_cls_results[0, i+track_num, :] ,track_query[0, i+track_num, :], tracker.tracks[i].id)          
                            index.append(i)
                            index.append(i + track_num)
                            track_id.append(tracker.tracks[-2].id)
                            track_id.append(tracker.tracks[-1].id)
                            tracked_index.append(len(tracker.tracks) -2)
                            tracked_index.append(len(tracker.tracks) -1)
                            # print("分裂：")
                            # print(tracker.tracks[i].id,(tracker.tracks[-2].id, tracker.tracks[-1].id))
                            # print(iou[i])
                            # print(center_distance[i])
                    elif track_cls_results[0, track_index[i], 0] >= t_o_thresh - 0.1 and track_box_results[0, track_index[i], 2] * track_box_results[0, track_index[i], 3] >= track_min_area:
                        if tracker.tracks[i].state==0:
                            index.append(i)
                            tracker.track_to_active(tracker.tracks[i], track_query[0, i, :], track_cls_results[0, i, :], track_box_results[0, i, :])
                            track_id.append(tracker.tracks[-1].id)
                            tracked_index.append(len(tracker.tracks) -1)
                        #正常追踪
                        else:
                            tracked_index.append(i)
                            index.append(track_index[i])
                            track_id.append(tracker.tracks[i].id)
                    else:
                        #追踪失败，
                        # un_track_index.append(track_index[i])
                        # print("丢失：")
                        # print(tracker.tracks[i].id)
                        # print(track_cls_results[0, track_index[i], 0])
                        # print(track_box_results[0, track_index[i]])
                        if (tracker.tracks[i].state == 0 and tracker.tracks[i].count_inactive >= tracker.inactive_patience) or tracker.tracks[i].state == 3:
                        #if  tracker.tracks[i].count_inactive >= tracker.inactive_patience or tracker.tracks[i].state == 3:
                            tracker.track_to_lost(tracker.tracks[i] ,tracker.frame_index)
                        elif tracker.tracks[i].state == 0:
                            tracker.tracks[i].count_inactive += 1
                        else:
                            tracker.track_to_inactive(tracker.tracks[i], tracker.frame_index)
                # 已追踪到的轨迹index
                index = torch.tensor(index, device=track_query.device).long()
                index = (torch.zeros_like(index, device=index.device, dtype=torch.int64), index)
                #追踪到的轨迹
                track_query = track_query[index].unsqueeze(0)
                track_box_results = track_box_results[index].unsqueeze(0)
                track_cls_results = track_cls_results[index].unsqueeze(0)
                track_pred_results = track_pred_results[index].unsqueeze(0)
                track_cls_results[:, :, 0] = 10.0 + track_cls_results[:, :, 0]
                track_id = torch.tensor(track_id, dtype=torch.int64)
                mask_cls_results = torch.cat([mask_cls_results[:, :self.num_queries], track_cls_results], dim=1)
                mask_box_results = torch.cat([mask_box_results[:, :self.num_queries], track_box_results], dim=1)
                mask_pred_results = torch.cat([mask_pred_results[:, :self.num_queries], track_pred_results], dim=1)
                mask_query = torch.cat([mask_query[:, :self.num_queries], track_query], dim=1)
            # upsample masks
            mask_center = None
            if self.crop:
                #每一个查询分割一小块区域,记录每个区域的中心点位置
                mask_pred_results = F.interpolate(
                    mask_pred_results,
                    size=(4*mask_pred_results.shape[-2], 4*mask_pred_results.shape[-1]),
                    mode="bilinear", 
                    align_corners=False,
                )
                mask_center = mask_box_results[:, :, :2]  #(B， Q， 2)
                mask_center = mask_center * torch.tensor([images.tensor.shape[-1], images.tensor.shape[-2]], device=mask_center.device)
                mask_center = mask_center - torch.tensor([padding_mask[0][0], padding_mask[0][1]], device=mask_center.device)
            else:
                mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
                )
                #去除填充部分
                mask_pred_results = mask_pred_results[:, :, padding_mask[0][1]:(images.tensor.shape[-2]-padding_mask[0][3]), padding_mask[0][0]:(images.tensor.shape[-1]-padding_mask[0][2])]
            del outputs
            i = 0
            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, query, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, mask_query, images.image_sizes
            ):  # image_size is augmented size, not divisible to 32
                height = input_per_image.get("height", image_size[0])  # real size
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                if self.crop:
                    new_size = image_size          
                else:
                    new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)

                all_id = torch.zeros(len(mask_box_result), dtype=torch.int64)
                all_id[self.num_queries:] = track_id

                if self.sem_seg_postprocess_before_inference:
                    # 还原至原来大小
                    if not self.crop:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, height, width
                        )
                    else:
                        mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                            mask_pred_result, image_size, int(height / (image_size[0]-padding_mask[0][1]-padding_mask[0][3]) * mask_pred_result.shape[-2]), int(width / (image_size[1]-padding_mask[0][0]-padding_mask[0][2]) * mask_pred_result.shape[-1])
                        )
                        mask_center = mask_center[0] * torch.tensor([width / (image_size[1]-padding_mask[0][0]-padding_mask[0][2]), height / (image_size[0]-padding_mask[0][1]-padding_mask[0][3])], device=mask_center.device)  #(q, 2)
                    mask_cls_result = mask_cls_result.to(mask_pred_result)
                    # mask_box_result = mask_box_result.to(mask_pred_result)
                    # mask_box_result = self.box_postprocess(mask_box_result, height, width)

                # semantic segmentation inference
                if self.semantic_on:
                    r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result)
                    if not self.sem_seg_postprocess_before_inference:
                        r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
                    processed_results[-1]["sem_seg"] = r

                # panoptic segmentation inference
                if self.panoptic_on:
                    panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
                    processed_results[-1]["panoptic_seg"] = panoptic_r

                # instance segmentation inference

                if self.instance_on:
                    mask_box_result = mask_box_result.to(mask_pred_result)
                    # height = new_size[0]/image_size[0]*height
                    # width = new_size[1]/image_size[1]*width
                    # mask_box_result_orgin = mask_box_result #归一化的边框
                    #TODO 边框的缩放未减去padding部分
                    mask_box_result = self.obbox_postprocess(mask_box_result, height, width) #(nq, 5)
                    if self.duplicate:
                        # instance_r = retry_if_cuda_oom(self.instance_inference_cell)(mask_cls_result, mask_pred_result, mask_box_result)
                        # instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                        instance_r, index, mask_box_result_return = retry_if_cuda_oom(self.instance_inference_nms)(mask_cls_result, mask_pred_result, mask_box_result)
                        #instance_i = retry_if_cuda_oom(self.instance_inference)(item_output["pred_logits"][i], item_output["pred_masks"][i], item_box_result)     
                    else:
                        #instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                        instance_r, index, mask_box_result_return = retry_if_cuda_oom(self.instance_inference_nms)(mask_cls_result, mask_pred_result, mask_box_result, tracker.detection_obj_score_thresh, tracker.detection_nms_thresh, (int(height), int(width)),self.crop, mask_center)
                        #instance_r = retry_if_cuda_oom(self.instance_inference_cell)(mask_cls_result, mask_pred_result, mask_box_result)
                        #instance_i = retry_if_cuda_oom(self.instance_inference)(item_output["pred_logits"][i], item_output["pred_masks"][i], item_box_result)     
                    #对追踪轨迹进行状态更新：
                    index_track = index[index >= self.num_queries] - self.num_queries
                    #由于box为归一化的边框，而在下一帧进行追踪时特征图是进行了padding的，所以需要将边框缩放到填充之后的大小，只需要对中心点进行缩放
                    mask_box_result_return[:, :2] = (mask_box_result_return[:, :2] * torch.tensor([width, height], device=mask_box_result_return.device) + torch.tensor([padding_mask[0][0], padding_mask[0][1]], device=mask_box_result_return.device)) / torch.tensor([width+padding_mask[0][0]+padding_mask[0][2], height+padding_mask[0][1]+padding_mask[0][3]], device=mask_box_result_return.device)
                    for i in range(len(tracked_index)):
                        if i in index_track :
                            tracker.tracks[tracked_index[i]].update(track_query[0, i, :], mask_box_result_return[i + self.num_queries, :], track_cls_results[0, i, :])
                        else :
                            # print("由于面积或IOU导致丢失:")
                            # print(tracker.tracks[tracked_index[i]].id)
                            # print(track_box_results[0, i, 2] * track_box_results[0, i, 3])
                            if (tracker.tracks[tracked_index[i]].state == 0 and tracker.tracks[tracked_index[i]].count_inactive >= tracker.inactive_patience): #patience=5
                            #if (tracker.tracks[tracked_index[i]].count_inactive >= tracker.inactive_patience) or tracker.tracks[tracked_index[i]].state == 3:  #patience=1
                                tracker.track_to_lost(tracker.tracks[tracked_index[i]] ,tracker.frame_index)
                            elif tracker.tracks[tracked_index[i]].state == 3:
                                if tracker.tracks[tracked_index[i]].start_frame == tracker.frame_index:
                                    tracker.tracks[tracked_index[i]].state = 2
                                else:
                                    tracker.track_to_lost(tracker.tracks[tracked_index[i]] ,tracker.frame_index)
                            elif tracker.tracks[tracked_index[i]].state == 0:
                                tracker.tracks[tracked_index[i]].count_inactive += 1
                            else:
                                tracker.track_to_inactive(tracker.tracks[tracked_index[i]], tracker.frame_index)
                    # 对于新生检测的判断
                    index_new = index[index < self.num_queries]
                    new_id = torch.arange(len(index_new)) + tracker.track_index
                    all_id[index_new] = new_id
                    if len(index_new) > 0:
                        tracker.add_tracks(mask_box_result_return[index_new], mask_cls_result[index_new].sigmoid(), query[index_new])
                    instance_r.track_id = all_id[index.to(all_id.device)]
                    # tracker.track_pos = torch.cat([mask_box_result[self.num_queries:], mask_box_result[index_new]], dim=0).unsqueeze(0)
                    # tracker.track_query = torch.cat([query[self.num_queries:], query[index_new]], dim=0).unsqueeze(0) 
                    tracker.step()   
                    processed_results[-1]["instances"] = instance_r
                    i += 1
            return processed_results
        

    def get_valid_id(self,scores,pred_masks,index):
        scores = scores
        masks = pred_masks
        track_ids = index + 1
        #根据scores对mask进行排序，对于重叠的mask，认为score的高的遮挡score低的mask
        _, index  = torch.sort(scores)

        seg = torch.zeros(pred_masks.shape[1:3],device=pred_masks.device,dtype=index.dtype)
    
        for i in index:
            if scores[i] > 0.01:
                mask = masks[i]
                seg[mask > 0] = track_ids[i]
        valid_index = []
        for i in track_ids:
            if torch.any(seg == i).item():
                valid_index.append(i-1)
        valid_index = torch.tensor(valid_index, device=index.device, dtype=index.dtype)
        return valid_index

    def get_valid_id_crop(self, scores, pred_masks, mask_size, mask_center, index):
        scores = scores
        masks = pred_masks
        track_ids = index
        #根据scores对mask进行排序，对于重叠的mask，认为score的高的遮挡score低的mask
        _, index  = torch.sort(scores)

        seg = torch.zeros(mask_size,device=pred_masks.device,dtype=index.dtype)
    
        for i in index:
            if scores[i] > 0.01:
                mask = masks[i]
                center = mask_center[i]
                start_x = int(center[0] - mask.shape[-1] / 2)
                start_y = int(center[1] - mask.shape[-2] / 2)
                end_x = int(start_x + mask.shape[-1])
                end_y = int(start_y + mask.shape[-2])
                # 确保区域不会超出大tensor的边界
                if start_x < 0:
                    start_x = 0
                    crop_x = mask.shape[-1] - end_x
                    mask = mask[:, crop_x:]
                if start_y < 0:
                    start_y = 0
                    crop_y = mask.shape[-2] - end_y
                    mask = mask[crop_y:, :]
                if end_x >  mask_size[1]:
                    end_x = mask_size[1]
                    crop_x = end_x - start_x
                    mask = mask[:, :crop_x]
                if end_y > mask_size[0]:
                    end_y = mask_size[0]
                    crop_y = end_y - start_y
                    mask = mask[:crop_y, :]
                seg[start_y:end_y, start_x:end_x][mask > 0] = track_ids[i]
        valid_index = []
        for i in track_ids:
            if torch.any(seg == i).item():
                valid_index.append(i)
        valid_index = torch.tensor(valid_index, device=index.device, dtype=index.dtype)
        return valid_index

    def prepare_targets(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets: 
            # pad gt
            h, w = targets_per_image.image_size
            image_size_xyxy = torch.as_tensor([w, h, w, h], dtype=torch.float, device=self.device)

            gt_masks = targets_per_image.gt_masks
            padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": padded_masks,
                    "boxes":box_ops.box_xyxy_to_cxcywh(targets_per_image.gt_boxes.tensor)/image_size_xyxy
                }
            )
        return new_targets


    def prepare_targets_crop(self, targets, images):
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size
            gt_masks = targets_per_image.gt_masks
            targets_per_image.gt_boxes.scale(1 / float(w), 1 / float(h))

            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": gt_masks,
                    "boxes": targets_per_image.gt_boxes.tensor,
                    "mask_pos": targets_per_image.gt_mask_pos,
                    "mask_size": targets_per_image.gt_mask_size[0],
                }
            )
        return new_targets


    def prepare_targets_detr(self, targets, images, moists, tracker):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        max_num = 0
        if tracker.track_ids is None:
            for targets_per_image, moist in zip(targets, moists):
            # pad gt
                h, w = targets_per_image.image_size
                gt_masks = targets_per_image.gt_masks
                gt_ids = targets_per_image.gt_ids
                if len(gt_ids) > max_num:
                    max_num = len(gt_ids)
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                targets_per_image.gt_boxes.scale(1 / float(w), 1 / float(h))
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "boxes": targets_per_image.gt_boxes.tensor,
                        "gt_id":targets_per_image.gt_ids,
                        "track_id": [],
                        "moist": moist,
                        "num_query": self.num_queries
                    }
                )
            tracker.max_num = max_num
        else:
            for targets_per_image, track_id, moist in zip(targets, tracker.track_ids, tracker.moists):
                # pad gt
                h, w = targets_per_image.image_size
                gt_masks = targets_per_image.gt_masks
                gt_ids = targets_per_image.gt_ids
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                targets_per_image.gt_boxes.scale(1 / float(w), 1 / float(h))
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "boxes": targets_per_image.gt_boxes.tensor,
                        "gt_id":targets_per_image.gt_ids,
                        "track_id": track_id,
                        "moist": moist,
                        "num_query": self.num_queries
                    }
                )
                #print(targets_per_image.gt_boxes.tensor.shape[0])
        return new_targets

    def prepare_targets_simi(self, targets, images, moists, tracker, norm_images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        max_num = 0
        if tracker.track_ids is None:
            for targets_per_image, moist, norm_image in zip(targets, moists, norm_images):
            # pad gt
                h, w = targets_per_image.image_size
                gt_masks = targets_per_image.gt_masks
                gt_ids = targets_per_image.gt_ids
                if len(gt_ids) > max_num:
                    max_num = len(gt_ids)
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                targets_per_image.gt_boxes.scale(1 / float(w), 1 / float(h))
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "boxes": targets_per_image.gt_boxes.tensor,
                        "gt_id":targets_per_image.gt_ids,
                        "track_id": [],
                        "moist": moist,
                        "num_query": self.num_queries,
                        "norm_image": norm_image
                    }
                )
            tracker.max_num = max_num
        else:
            for targets_per_image, track_id, moist, interior_img, exterior_img, interior_fea, exterior_fea, last_box, norm_image, pixel in zip(targets, tracker.track_ids, tracker.moists, tracker.interior_img, tracker.exterior_img, tracker.interior_fea, tracker.exterior_fea, tracker.last_boxes, norm_images, tracker.pixel_num):
                # pad gt
                h, w = targets_per_image.image_size
                gt_masks = targets_per_image.gt_masks
                gt_ids = targets_per_image.gt_ids
                padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
                padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
                targets_per_image.gt_boxes.scale(1 / float(w), 1 / float(h))
                new_targets.append(
                    {
                        "labels": targets_per_image.gt_classes,
                        "masks": padded_masks,
                        "boxes": targets_per_image.gt_boxes.tensor,
                        "gt_id":targets_per_image.gt_ids,
                        "track_id": track_id,
                        "moist": moist,
                        "num_query": self.num_queries,
                        "interior_img": interior_img,
                        "exterior_img": exterior_img,
                        "interior_fea": interior_fea,
                        "exterior_fea": exterior_fea,
                        "last_boxes": last_box,
                        "norm_image": norm_image,
                        "pixel_num": pixel,
                    }
                )
        return new_targets

    def prepare_targets_live_cell(self, targets, images):
        h_pad, w_pad = images.tensor.shape[-2:]
        new_targets = []
        for targets_per_image in targets:
            # pad gt
            h, w = targets_per_image.image_size

            # padded_masks = torch.zeros((gt_masks.shape[0], h_pad, w_pad), dtype=gt_masks.dtype, device=gt_masks.device)
            # padded_masks[:, : gt_masks.shape[1], : gt_masks.shape[2]] = gt_masks
            targets_per_image.gt_boxes.scale(1 / float(w), 1 / float(h))
            new_targets.append(
                {
                    "labels": targets_per_image.gt_classes,
                    "masks": targets_per_image.gt_masks,
                    "boxes": targets_per_image.gt_boxes.tensor
                }
            )
        return new_targets

    def semantic_inference(self, mask_cls, mask_pred):
        # if use cross-entropy loss in training, evaluate with softmax
        if self.semantic_ce_loss:
            mask_cls = F.softmax(mask_cls, dim=-1)[..., :-1]
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg
        # if use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        else:
            T = self.pano_temp
            mask_cls = mask_cls.sigmoid()
            if self.transform_eval:
                mask_cls = F.softmax(mask_cls / T, dim=-1)  # already sigmoid
            mask_pred = mask_pred.sigmoid()
            semseg = torch.einsum("qc,qhw->chw", mask_cls, mask_pred)
            return semseg

    def panoptic_inference(self, mask_cls, mask_pred):
        # As we use focal loss in training, evaluate with sigmoid. As sigmoid is mainly for detection and not sharp
        # enough for semantic and panoptic segmentation, we additionally use use softmax with a temperature to
        # make the score sharper.
        prob = 0.5
        T = self.pano_temp
        scores, labels = mask_cls.sigmoid().max(-1)
        mask_pred = mask_pred.sigmoid()
        keep = labels.ne(self.sem_seg_head.num_classes) & (scores > self.object_mask_threshold)
        # added process
        if self.transform_eval:
            scores, labels = F.softmax(mask_cls.sigmoid() / T, dim=-1).max(-1)
        cur_scores = scores[keep]
        cur_classes = labels[keep]
        cur_masks = mask_pred[keep]
        cur_prob_masks = cur_scores.view(-1, 1, 1) * cur_masks

        h, w = cur_masks.shape[-2:]
        panoptic_seg = torch.zeros((h, w), dtype=torch.int32, device=cur_masks.device)
        segments_info = []

        current_segment_id = 0

        if cur_masks.shape[0] == 0:
            # We didn't detect any mask :(
            return panoptic_seg, segments_info
        else:
            # take argmax
            cur_mask_ids = cur_prob_masks.argmax(0)
            stuff_memory_list = {}
            for k in range(cur_classes.shape[0]):
                pred_class = cur_classes[k].item()
                isthing = pred_class in self.metadata.thing_dataset_id_to_contiguous_id.values()
                mask_area = (cur_mask_ids == k).sum().item()
                original_area = (cur_masks[k] >= prob).sum().item()
                mask = (cur_mask_ids == k) & (cur_masks[k] >= prob)

                if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                    if mask_area / original_area < self.overlap_threshold:
                        continue

                    # merge stuff regions
                    if not isthing:
                        if int(pred_class) in stuff_memory_list.keys():
                            panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                            continue
                        else:
                            stuff_memory_list[int(pred_class)] = current_segment_id + 1

                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id

                    segments_info.append(
                        {
                            "id": current_segment_id,
                            "isthing": bool(isthing),
                            "category_id": int(pred_class),
                        }
                    )

            return panoptic_seg, segments_info

    def instance_inference(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        labels = torch.arange(self.sem_seg_head.num_classes, device=self.device).unsqueeze(0).repeat(self.num_queries, 1).flatten(0, 1)
        #scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        n = scores.shape[0]
        scores_per_image, topk_indices = scores.flatten(0, 1).topk(n , sorted=False)  # select all
        labels_per_image = labels[topk_indices]
        topk_indices = topk_indices // self.sem_seg_head.num_classes
        mask_pred = mask_pred[topk_indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = BitMasks_ct(result.pred_masks).get_oriented_bounding_boxes(norm = False).to(mask_pred.device) 
        #mask_box_result = mask_box_result[topk_indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = RotatedBoxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def instance_inference_nms(self, mask_cls, mask_pred, mask_box_result, score_thresd, nms_thresd, image_size, crop=False, mask_center=None):
        # mask_pred is already processed to have the same shape as original input
        #image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        #scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        #将box scors与mask scores相乘
        pred_masks = (mask_pred > 0).float()
        mask_scores = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)
        #track_box_result = mask_box_result[self.num_queries:]
        scores[:, 0] = mask_scores * scores[:, 0]
        #print(scores[self.num_queries:, 0])
        if crop:
            mask_box_result = BitMasks_ct(pred_masks).get_oriented_bounding_boxes_crop(mask_center, norm = False).to(pred_masks.device)
        else:
            mask_box_result = BitMasks_ct(pred_masks).get_oriented_bounding_boxes(norm = False).to(pred_masks.device)  
        #track_box_result = mask_box_result[self.num_queries:]
        angle_scale = torch.tensor([1,1,1,1,90], device=mask_box_result.device)
        mask_box_result_return = mask_box_result / angle_scale
        mask_box_result_return = box_ops.scale_obbox(mask_box_result_return, torch.tensor([1/image_size[1], 1/image_size[0]], dtype=float, device=mask_box_result_return.device), norm_angle= True)
        if len(scores) == self.num_queries: #第一帧检测，加大阈值过滤掉由于旋转框IOU计算错误导致的检测
            nms_thresd = 0.3
            score_thresd = score_thresd - 0.1
        mask_box_result_per_image, scores_per_image, labels_per_image, index = box_ops.multiclass_nms_rotated(mask_box_result.squeeze(0), scores.squeeze(0), score_thresd, nms_thresd, return_inds=True) 
        # 过滤掉有小面积较小的检测和利用中心点距离来处理IOU没有过滤掉的错误检测：
        valid_index = []
        index_track = index[index >= self.num_queries] 
        if len(index_track) > 0:
            index_new = index[index < self.num_queries]
            det_distance = torch.cdist(mask_box_result_return[index_new, :2], mask_box_result_return[index_track, :2])
            min_values, _ = torch.min(det_distance, dim=1)
            index_new = index_new[min_values > 0.01]
            index = torch.cat((index_track, index_new), dim=0)
        for i in index:
            if mask_box_result_return[i][2] * mask_box_result_return[i][3] > 0.00001:
                valid_index.append(i)
        index = torch.tensor(valid_index, device=index.device)
        #根据mask中的相互遮挡判断
        if self.crop:
            index = self.get_valid_id_crop(scores[:,0][index], pred_masks[index], image_size, mask_center[index], index)
        else:
            index = self.get_valid_id(scores[:,0][index], pred_masks[index], index)
        mask_box_result = mask_box_result[index]
        scores_per_image = scores[:, 0][index]
        labels_per_image = torch.zeros_like(index, device=labels_per_image.device)

        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = pred_masks[index]
        # half mask box half pred box
        # mask_box_result = BitMasks_ct(result.pred_masks).get_oriented_bounding_boxes(norm = False).to(pred_masks.device)  
        result.pred_boxes = RotatedBoxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()
        # calculate average mask prob
        # mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        # if self.focus_on_box:
        #     mask_scores_per_image = 1.0
        #result.scores = scores_per_image * mask_scores_per_image
        result.scores = scores_per_image
        result.pred_classes = labels_per_image
        return result, index, mask_box_result_return


    def instance_inference_cell(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        #indices =  torch.where((scores[:, 0] > scores[:, 1]) & (scores[:, 0] > 0.2))[0]
        indices =  torch.where((scores[:, 0] > 0.3))[0]
        scores_per_image = scores[indices][:, 0]
        labels_per_image = torch.zeros_like(indices)
        mask_pred = mask_pred[indices]
        # if this is panoptic segmentation, we only keep the "thing" classes
        if self.panoptic_on:
            keep = torch.zeros_like(scores_per_image).bool()
            for i, lab in enumerate(labels_per_image):
                keep[i] = lab in self.metadata.thing_dataset_id_to_contiguous_id.values()
            scores_per_image = scores_per_image[keep]
            labels_per_image = labels_per_image[keep]
            mask_pred = mask_pred[keep]
        result = Instances(image_size)
        # mask (before sigmoid)
        result.pred_masks = (mask_pred > 0).float()
        # half mask box half pred box
        mask_box_result = mask_box_result[indices]
        if self.panoptic_on:
            mask_box_result = mask_box_result[keep]
        result.pred_boxes = RotatedBoxes(mask_box_result)
        # Uncomment the following to get boxes from masks (this is slow)
        # result.pred_boxes = BitMasks(mask_pred > 0).get_bounding_boxes()

        # calculate average mask prob
        mask_scores_per_image = (mask_pred.sigmoid().flatten(1) * result.pred_masks.flatten(1)).sum(1) / (result.pred_masks.flatten(1).sum(1) + 1e-6)
        if self.focus_on_box:
            mask_scores_per_image = 1.0
        result.scores = scores_per_image * mask_scores_per_image
        result.pred_classes = labels_per_image
        return result

    def box_postprocess(self, out_bbox, img_h, img_w):
        # postprocess box height and width
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([img_w, img_h, img_w, img_h])
        scale_fct = scale_fct.to(out_bbox)
        boxes = boxes * scale_fct
        return boxes

    def obbox_postprocess(self, out_obbox, img_h, img_w):
        scale_fct = torch.tensor([img_w, img_h])
        scale_fct = scale_fct.to(out_obbox)
        obboxes = box_ops.scale_obbox(out_obbox, scale_fct, norm_angle=True)
        obboxes[:, 4] *= 90
        return obboxes

from collections import deque
import torch



class Tracker:
    def __init__(self, detection_obj_score_thresh = 0.3, 
        track_obj_score_thresh = 0.2,
        detection_nms_thresh = 0.4,
        track_nms_thresh = 0.15,
        public_detections = None,
        inactive_patience = 3,
        logger = None,
        detection_query_num = None,
        dn_query_num = None,
        ):
        
        self.detection_obj_score_thresh = detection_obj_score_thresh
        self.track_obj_score_thresh = track_obj_score_thresh
        self.detection_nms_thresh = detection_nms_thresh
        self.track_nms_thresh = track_nms_thresh

        self.public_detections = public_detections
        self.inactive_patience = inactive_patience
        self.detection_query_num = detection_query_num
        self.dn_query_num = dn_query_num
        self._logger = logger
        if self._logger is None:
            self._logger = lambda *log_strs: None
        #training use
        self.track_ids = None
        self.track_index = 0 
        self.track_pos = None
        self.track_query = None
        self.moists = None
        #self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
        #self.reid_sim_only = tracker_cfg['reid_sim_only']
        #self.generate_attention_maps = generate_attention_maps
        #self.reid_score_thresh = tracker_cfg['reid_score_thresh']
        #self.reid_greedy_matching = tracker_cfg['reid_greedy_matching']
        #self.prev_frame_dist = tracker_cfg['prev_frame_dist']
        #self.steps_termination = tracker_cfg['steps_termination']

    def reset(self, hard=True):
        self.tracks = []
        self.inactive_tracks = []
        #self._prev_features = deque([None], maxlen=self.prev_frame_dist)
        self.track_ids = None
        self.moists = None
        self.track_index = 0
        self.track_num = 0
        self.results = {}
        self.frame_index = 0

    def track_to_lost(self, track, frame):
        track.lost(frame)
        #self.inactive_tracks -= track

    def track_to_inactive(self, track, frame):
        track.inactive(frame)
        #self.inactive_tracks += track

    def tracks_to_inactive(self, tracks):
        self.tracks = [t for t in self.tracks if t not in tracks]

        for track in tracks:
            track.pos = track.last_pos[-1]
        self.inactive_tracks += tracks

    def add_track(self, pos, scores, query_embeds, mather_id=None):
        self.tracks.append(Track(
            query_embeds,
            pos,
            scores,
            self.track_index,
            self.frame_index,
            mather_id,
        )
        )
        self.track_index += 1

    def add_tracks(self, pos, scores, query_embeds, masks=None, attention_maps=None, aux_results=None):
        """Initializes new Track objects and saves them."""
        for i in range(len(pos)):
            self.tracks.append(Track(
                query_embeds[i],
                pos[i],
                scores[i],
                self.track_index + i,
                self.frame_index,
            ))
        self.track_index += len(pos)

    
    def step(self):
    #     """This function should be called every timestep to perform tracking with a blob
    #     containing the image information.
    #     """
    #    inactive_tracks = []
        tracks = []
        track_pos = []
        track_query = []
        for track in self.tracks:
            if track.has_positive_area and track.state != 2:
                tracks.append(track)
                track_pos.append(track.pos.unsqueeze(0))
                track_query.append(track.query_emb.unsqueeze(0))
        self.track_pos = torch.cat(track_pos, dim=0).unsqueeze(0)
        self.track_query = torch.cat(track_query, dim=0).unsqueeze(0)
        self.tracks = tracks
        self.track_num = len(self.tracks)
        self.frame_index += 1



class Track(object):
    """This class contains all necessary for every individual track."""

    def __init__(self, query_emb, pos, score, track_id, start_frame=0, obj_ind=None,
                 mather_id=None, mask=None, attention_map=None):
        self.id = track_id   #轨迹id
        self.query_emb = query_emb  #query
        self.pos = pos #位置编码
        self.last_pos = deque([pos.clone()]) #追踪丢失前最后的位置
        self.score = score #预测分数
        self.ims = deque([])
        self.count_inactive = 0  #丢失帧数
        self.count_termination = 0
        self.gt_id = None #训练时对应的真实样本id
        self.obj_ind = obj_ind
        self.mather_id = mather_id #母细胞id
        self.mask = mask
        self.attention_map = attention_map
        self.state = 3   #0代表inactive, 1代表track, 2表示lost， 3表示刚被检测 
        self.start_frame = start_frame
        self.end_frame = None

    def update(self, query_emb, pos, score):
        self.query_emb = query_emb
        self.pos = pos
        self.score = score
        self.state = 1
        self.count_inactive = 0

    def moist(self, query_emb, pos, score, id):
        self.father_id = self.id
        self.query_emb = query_emb
        self.score = score 
        self.pos = pos
        self.id = id 

    def inactive(self, frame):
        self.count_inactive += 1
        self.state = 0
        self.end_frame = frame

    def lost(self, frame):
        self.state = 2
        self.end_frame = frame

    def has_positive_area(self) -> bool:
        """Checks if the current position of the track has
           a valid, .i.e., positive area, bounding box."""
        return self.pos[2] * self.pos[3] > 0.001

    def reset_last_pos(self) -> None:
        """Reset last_pos to the current position of the track."""
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())

