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
        #mask = [x["padding_mask"].to(self.device) for x in batched_inputs]

        features = self.backbone(images.tensor)

        if self.training:
            # dn_args={"scalar":30,"noise_scale":0.4}
            # mask classification target
            if "instances" in batched_inputs[0]:
                gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
                moists = [x["moists"] for x in batched_inputs]
                if "live_cell" in self.data_loader:
                     targets = self.prepare_targets_live_cell(gt_instances, images)
                elif 'detr' in self.data_loader:
                    if self.crop:
                        targets = self.prepare_targets_crop(gt_instances, images)
                    else:
                        targets = self.prepare_targets_detr(gt_instances, images, moists, tracker) #对边框进行归一化，角度没有进行归一化
                else:
                    targets = self.prepare_targets(gt_instances, images)
            else:
                targets = None
            outputs, mask_dict = self.sem_seg_head(features ,targets=targets, tracker=tracker)
            # bipartite matching-based loss
            losses = self.criterion(outputs, targets, mask_dict, is_track=True, tracker=tracker)

            for k in list(losses.keys()):
                if k in self.criterion.weight_dict:
                    losses[k] *= self.criterion.weight_dict[k]
                else:
                    # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
            return losses
        else:
            outputs, _ = self.sem_seg_head(features, tracker=tracker)
            mask_cls_results = outputs["pred_logits"]
            mask_pred_results = outputs["pred_masks"]
            mask_box_results = outputs["pred_boxes"]
            mask_query = outputs["query"]
            # test
            item_output = outputs["interm_outputs"]
            track_id = torch.tensor([], dtype=torch.int64)
            # 对追踪query进行处理
            if tracker.track_num > 0:
                t_o_thresh = tracker.track_obj_score_thresh
                track_num = tracker.track_num
                track_cls_results = mask_cls_results[:, self.num_queries:, :].sigmoid()
                track_box_results = mask_box_results[:, self.num_queries:, :]
                track_pred_results = mask_pred_results[:, self.num_queries:, :, :]
                index = []
                un_track_index = []  #首次检测到的追踪轨迹
                track_id = []
                track_query = mask_query[:, self.num_queries:, :]
                iou = 1 - kfiou_loss(track_box_results[0, :track_num, :], track_box_results[0, track_num:, :])
                max_index = track_cls_results[:, :track_num, 0] - track_cls_results[:, track_num:, 0]
                max_index = max_index.squeeze(0)
                max_index[max_index > 0] = 1
                max_index[max_index < 0] = 0
                track_index = torch.arange(0, track_num).to(max_index.device)
                track_index = track_index + max_index * track_num
                track_index = track_index.long()
                # angle_convert = torch.tensor([1.0, 1.0, 1.0, 1.0, math.pi / 2.0], device= mask_box_results.device)
                for i in range(track_num):
                    if track_cls_results[0, i, 0] > t_o_thresh and track_cls_results[0, i + track_num, 0] > t_o_thresh:
                        #未发生分裂
                        #tracker.track_nms_thresh = 0.5
                        if iou[i] > tracker.track_nms_thresh:
                            tracker.tracks[i].update(track_query[0, track_index[i], :], track_box_results[0, track_index[i], :], track_cls_results[0, track_index[i], :])
                            index.append(track_index[i])
                            track_id.append(tracker.tracks[i].id)
                        #发生分裂
                        else:
                            if track_box_results[0, i, 2] * track_box_results[0][i][3] > 0.005 and track_box_results[0, i+track_num, 2] * track_box_results[0][i+track_num][3] > 0.005:
                                tracker.tracks[i].lost(tracker.frame_index)
                                tracker.add_track(track_box_results[0, i, :], track_cls_results[0, i, :] ,track_query[0, i, :], tracker.tracks[i].id)
                                tracker.add_track(track_box_results[0, i+track_num, :], track_cls_results[0, i+track_num, :] ,track_query[0, i+track_num, :], tracker.tracks[i].id)          
                                index.append(i)
                                index.append(i + track_num)
                                track_id.append(tracker.tracks[-2].id)
                                track_id.append(tracker.tracks[-1].id)
                                print(tracker.tracks[i].id,(tracker.tracks[-2].id, tracker.tracks[-1].id))
                            else:
                                tracker.tracks[i].update(track_query[0, track_index[i], :], track_box_results[0, track_index[i], :], track_cls_results[0, track_index[i], :])
                                index.append(track_index[i])
                                track_id.append(tracker.tracks[i].id)
                    elif track_cls_results[0, track_index[i], 0] >= t_o_thresh:
                        #正常追踪
                        tracker.tracks[i].update(track_query[0, track_index[i], :], track_box_results[0, track_index[i], :], track_cls_results[0, track_index[i], :])
                        index.append(track_index[i])
                        track_id.append(tracker.tracks[i].id)
                    else:
                        #追踪失败，
                        # un_track_index.append(track_index[i])
                        if (tracker.tracks[i].state == 0 and tracker.tracks[i].count_inactive >= tracker.inactive_patience) or tracker.tracks[i].state == 3:
                            tracker.track_to_lost(tracker.tracks[i] ,tracker.frame_index)
                        elif tracker.tracks[i].state == 0:
                            tracker.tracks[i].count_inactive += 1
                        else:
                            tracker.track_to_inactive(tracker.tracks[i], tracker.frame_index)
                # 已追踪到的轨迹index
                index = torch.tensor(index, device=track_query.device).long()
                index = (torch.zeros_like(index, device=index.device, dtype=torch.int64), index)
                # 未追踪到的轨迹index
                # un_track_index = torch.tensor(un_track_index, device=track_query.device).long()
                # un_track_index = (torch.zeros_like(un_track_index, device=un_track_index.device, dtype=torch.int64), un_track_index)
                #未追踪到的轨迹
                # un_track_box_results = track_box_results[un_track_index]
                # un_track_query = track_query[un_track_index]
                # un_track_cls_results = track_cls_results[un_track_index]
                #追踪到的轨迹
                track_query = track_query[index].unsqueeze(0)
                track_box_results = track_box_results[index].unsqueeze(0)
                track_cls_results = track_cls_results[index].unsqueeze(0)
                track_pred_results = track_pred_results[index].unsqueeze(0)
                track_cls_results[:, :, 0] = 1
                track_id = torch.tensor(track_id, dtype=torch.int64)
                mask_cls_results = torch.cat([mask_cls_results[:, :self.num_queries], track_cls_results], dim=1)
                mask_box_results = torch.cat([mask_box_results[:, :self.num_queries], track_box_results], dim=1)
                mask_pred_results = torch.cat([mask_pred_results[:, :self.num_queries], track_pred_results], dim=1)
                mask_query = torch.cat([mask_query[:, :self.num_queries], track_query], dim=1)
            # upsample masks
            mask_pred_results = F.interpolate(
                mask_pred_results,
                size=(images.tensor.shape[-2], images.tensor.shape[-1]),
                mode="bilinear",
                align_corners=False,
            )
            # item_output["pred_masks"] = F.interpolate(
            #     item_output["pred_masks"],
            #     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
            #     mode="bilinear",
            #     align_corners=False,
            # )
            del outputs
            i = 0
            processed_results = []
            for mask_cls_result, mask_pred_result, mask_box_result, input_per_image, query, image_size in zip(
                mask_cls_results, mask_pred_results, mask_box_results, batched_inputs, mask_query, images.image_sizes
            ):  # image_size is augmented size, not divisible to 32
                height = input_per_image.get("height", image_size[0])  # real size
                width = input_per_image.get("width", image_size[1])
                processed_results.append({})
                new_size = mask_pred_result.shape[-2:]  # padded size (divisible to 32)

                all_id = torch.zeros(len(mask_box_result), dtype=torch.int64)
                all_id[self.num_queries:] = track_id

                if self.sem_seg_postprocess_before_inference:
                    # 还原至原来大小
                    mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
                        mask_pred_result, image_size, height, width
                    )
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
                    height = new_size[0]/image_size[0]*height
                    width = new_size[1]/image_size[1]*width
                    # mask_box_result_orgin = mask_box_result #归一化的边框
                    mask_box_result = self.obbox_postprocess(mask_box_result, height, width) #(nq, 5)
                    # item_box_result = self.obbox_postprocess(item_output["pred_boxes"][i], height, width) #(nq, 5)
                    if self.duplicate:
                        # instance_r = retry_if_cuda_oom(self.instance_inference_cell)(mask_cls_result, mask_pred_result, mask_box_result)
                        # instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                        instance_r, index, mask_box_result_return = retry_if_cuda_oom(self.instance_inference_nms)(mask_cls_result, mask_pred_result, mask_box_result)
                        #instance_i = retry_if_cuda_oom(self.instance_inference)(item_output["pred_logits"][i], item_output["pred_masks"][i], item_box_result)     
                    else:
                        #instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, mask_box_result)
                        instance_r, index, mask_box_result_return = retry_if_cuda_oom(self.instance_inference_nms)(mask_cls_result, mask_pred_result, mask_box_result)
                        #instance_r = retry_if_cuda_oom(self.instance_inference_cell)(mask_cls_result, mask_pred_result, mask_box_result)
                        # instance_i = retry_if_cuda_oom(self.instance_inference)(item_output["pred_logits"][i], item_output["pred_masks"][i], item_box_result)     
                    # 对于新生检测的判断
                    index_new = index[index < self.num_queries]
                    # 又重新使用了基于IOU的匹配方法，违背端到端的初衷
                    # if tracker.track_num > 0 and len(un_track_box_results) > 0: 
                    #     new_track_box = mask_box_result_return[index_new]                   
                    #     angle_convert = torch.tensor([1.0, 1.0, 1.0, 1.0, math.pi / 2.0], device= new_track_box.device)
                    #     iou = box_iou_rotated(new_track_box.float() * angle_convert, un_track_box_results.float() * angle_convert)
                    #     max_value, max_index = torch.max(iou, dim=1)
                    #     for i in range(len(iou)):
                    #         if max_value[i] > 0.6:
                    #             tracker.tracks[un_track_index[max_index[i]]].update(query[index_new[i]], mask_box_result_return[index_new[i], :], mask_cls_result[index_new].sigmoid())
                    #             all_id[index_new[i]] = tracker.tracks[un_track_index[max_index[i]]].id
                    #         else:                
                    #             tracker.add_track(mask_box_result_return[index_new[i]], mask_cls_result[index_new[i]].sigmoid(), query[index_new[i]])
                    #             all_id[index_new[i]] = tracker.tracks[-1].id
                    # else:                
                    new_id = torch.arange(len(index_new)) + tracker.track_index
                    all_id[index_new] = new_id
                    if len(index_new) > 0:
                        tracker.add_tracks(mask_box_result_return[index_new], mask_cls_result[index_new].sigmoid(), query[index_new])
                    instance_r.track_id = all_id[index]
                    # tracker.track_pos = torch.cat([mask_box_result[self.num_queries:], mask_box_result[index_new]], dim=0).unsqueeze(0)
                    # tracker.track_query = torch.cat([query[self.num_queries:], query[index_new]], dim=0).unsqueeze(0) 
                    tracker.step()   
                    processed_results[-1]["instances"] = instance_r
                    # processed_results[-1]["item_instances"] = instance_i
                    # processed_results[-1]["item_box"] = item_box_result
                    i += 1
            return processed_results

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

    def instance_inference_nms(self, mask_cls, mask_pred, mask_box_result):
        # mask_pred is already processed to have the same shape as original input
        image_size = mask_pred.shape[-2:]
        scores = mask_cls.sigmoid()  # [100, 80]
        #scores_per_image, topk_indices = scores.flatten(0, 1).topk(self.test_topk_per_image, sorted=False)  # select 100
        #将box scors与mask scores相乘
        pred_masks = (mask_pred > 0).float()
        mask_scores = (mask_pred.sigmoid().flatten(1) * pred_masks.flatten(1)).sum(1) / (pred_masks.flatten(1).sum(1) + 1e-6)
        #track_mask_scores = mask_scores[self.num_queries:]
        track_box_result = mask_box_result[self.num_queries:]
        scores[:, 0] = mask_scores * scores[:, 0]
        mask_box_result = BitMasks_ct(pred_masks).get_oriented_bounding_boxes(norm = False).to(pred_masks.device)  
        track_box_result = mask_box_result[self.num_queries:]
        angle_scale = torch.tensor([1,1,1,1,90], device=mask_box_result.device)
        mask_box_result_return = mask_box_result / angle_scale
        mask_box_result_return = box_ops.scale_obbox(mask_box_result_return, torch.tensor([1/image_size[0], 1/image_size[1]], dtype=float, device=mask_box_result_return.device), norm_angle= True)
        mask_box_result_per_image, scores_per_image, labels_per_image, index = box_ops.multiclass_nms_rotated(mask_box_result.squeeze(0), scores.squeeze(0), 0.2, 0.4, return_inds=True) 
        # 过滤掉有小面积较小的检测：
        valid_index = []
        for i in index:
            if mask_box_result_return[i][2] * mask_box_result_return[i][3] > 0.001:
                valid_index.append(i)
        index = torch.tensor(valid_index, device=index.device)
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
        track_nms_thresh = 0.3,
        public_detections = None,
        inactive_patience = 5,
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

