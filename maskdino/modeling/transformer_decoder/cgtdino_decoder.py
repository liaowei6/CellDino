# ------------------------------------------------------------------------
# DINO
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
import logging
import fvcore.nn.weight_init as weight_init
import torch
import math
import cv2
import numpy as np
import copy
import random
from torch import nn, Tensor
from ..pixel_decoder.ops.modules import MSDeformAttn
from torch.nn import functional as F
from torch.cuda.amp import autocast

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap

from detectron2.config import configurable
from detectron2.layers import Conv2d
from detectron2.utils.registry import Registry
from detectron2.structures import BitMasks
from .ctdino_decoder import TRANSFORMER_DECODER_REGISTRY
from typing import Optional, List, Union
#from .dino_decoder import TransformerDecoder, DeformableTransformerDecoderLayer
from ...utils.utils import MLP, gen_encoder_output_proposals_ct, inverse_sigmoid, BitMasks_ct, _get_clones, _get_activation_fn, gen_sineembed_for_position, get_rotated_rect_vertices, extract_region_features
from ...utils import box_ops
from ..pixel_decoder.position_encoding import PositionEmbeddingSine
from .cmm import CategoricalCounting
from .cgfe import CGFE, MultiScaleFeature
# TRANSFORMER_DECODER_REGISTRY = Registry("TRANSFORMER_MODULE")
# TRANSFORMER_DECODER_REGISTRY.__doc__ = """
# Registry for transformer module in CgtDINO.
# """

def build_transformer_decoder(cfg, in_channels, mask_classification=True):
    """
    Build a instance embedding branch from `cfg.MODEL.INS_EMBED_HEAD.NAME`.
    """
    name = cfg.MODEL.MaskDINO.TRANSFORMER_DECODER_NAME
    return TRANSFORMER_DECODER_REGISTRY.get(name)(cfg, in_channels, mask_classification)



def nms_distance(center_points, scores, distance_threshold):
        # 计算中心点之间的距离矩阵
        distances = torch.cdist(center_points, center_points)
    
        # 将对角线上的元素设为无穷大，以避免将中心点与自身进行比较
        diagonal = torch.eye(center_points.size(0), device=distances.device).bool()
        distances.masked_fill_(diagonal, float('inf'))
    
        # 找到保留的索引
        keep = []
        while torch.nonzero(scores).size(0) > 0:
            max_idx = torch.argmax(scores)
            keep.append(max_idx.item())
            overlap = distances[max_idx] < distance_threshold
            scores[overlap] = 0
            scores[max_idx] = 0
    
        return keep

@TRANSFORMER_DECODER_REGISTRY.register()
class CgtDINODecoder(nn.Module):
    @configurable
    def __init__(
            self,
            in_channels,
            mask_classification=True,
            *,
            num_classes: int,
            hidden_dim: int,
            num_queries: int,
            nheads: int,
            dim_feedforward: int,
            dec_layers: int,
            mask_dim: int,
            enforce_input_project: bool,
            two_stage: bool,
            dn: str,
            noise_scale:float,
            dn_num:int,
            initialize_box_type:bool,
            initial_pred:bool,
            learn_tgt: bool,
            total_num_feature_levels: int = 5,
            dropout: float = 0.0,
            activation: str = 'relu',
            nhead: int = 8,
            dec_n_points: int = 4,
            return_intermediate_dec: bool = True,
            query_dim: int = 5,
            dec_layer_share: bool = False,
            semantic_ce_loss: bool = False,
            with_position: bool = False,
            nms_query_select: bool = False,
            dn_aug: bool = False,
            temperature: int = 100000,
            crop: False,
            new_pos: False,
            simi: False,
            dq: False,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            in_channels: channels of the input features
            mask_classification: whether to add mask classifier or not
            num_classes: number of classes
            hidden_dim: Transformer feature dimension
            num_queries: number of queries
            nheads: number of heads
            dim_feedforward: feature dimension in feedforward network
            enc_layers: number of Transformer encoder layers
            dec_layers: number of Transformer decoder layers
            pre_norm: whether to use pre-LayerNorm or not
            mask_dim: mask feature dimension
            enforce_input_project: add input project 1x1 conv even if input
                channels and hidden dim is identical
            d_model: transformer dimension
            dropout: dropout rate
            activation: activation function
            nhead: num heads in multi-head attention
            dec_n_points: number of sampling points in decoder
            return_intermediate_dec: return the intermediate results of decoder
            query_dim: 5 -> (x, y, a, b, \theta)
            dec_layer_share: whether to share each decoder layer
            semantic_ce_loss: use ce loss for semantic segmentation
        """
        super().__init__()

        assert mask_classification, "Only support mask classification model"
        self.mask_classification = mask_classification
        self.num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred

        # define Transformer decoder here
        self.dn=dn
        self.learn_tgt = learn_tgt
        self.noise_scale=noise_scale
        self.dn_num=dn_num
        self.dn_aug = dn_aug
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage=two_stage
        self.initialize_box_type = initialize_box_type
        self.total_num_feature_levels = total_num_feature_levels
        
        self.num_queries = num_queries
        self.topk = num_queries
        self.semantic_ce_loss = semantic_ce_loss
        # learnable query features
        if not two_stage or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if not two_stage and initialize_box_type == 'no':
            self.query_embed = nn.Embedding(num_queries, 5)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(in_channels, hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())
        self.num_classes=num_classes
        # output FFNs
        assert self.mask_classification, "why not class embedding?"
        if self.mask_classification:
            if self.semantic_ce_loss:
                self.class_embed = nn.Linear(hidden_dim, num_classes+1)
            else:
                self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc=nn.Embedding(num_classes,hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        self.mask_embed_item = MLP(hidden_dim, hidden_dim, mask_dim, 3)
        #运动预测
        # self.motion_cls_pred = nn.Linear(hidden_dim, 1)
        # self.motion_pred = MLP(hidden_dim, hidden_dim, 5, 3)
        # self.motion_pred_first = nn.Linear(hidden_dim, 2)
        # self.motion_pred_second = nn.Linear(hidden_dim, 2)
        # init decoder
        self.decoder_norm = decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points)
        self.decoder = TransformerDecoder(decoder_layer, self.num_layers, decoder_norm,
                                          return_intermediate=return_intermediate_dec,
                                          d_model=hidden_dim, query_dim=query_dim,
                                          num_feature_levels=self.num_feature_levels,
                                          dec_layer_share=dec_layer_share,
                                          temperature=temperature,
                                          new_pos=new_pos,
                                          )
        # 对追踪query进行预处理
        # self.pre_track_query_process = MLP(hidden_dim, hidden_dim, hidden_dim, 3)
        # self.track_pos_pred = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
        #                                                   dropout, activation,
        #                                                   self.num_feature_levels, nhead, dec_n_points, motion_pred=True)
        # self.detection_query_process = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
        #                                                   dropout, activation,
        #                                                   self.num_feature_levels, nhead, dec_n_points)
        self.track_query_process = DeformableTransformerDecoderLayer(hidden_dim, dim_feedforward,
                                                          dropout, activation,
                                                          self.num_feature_levels, nhead, dec_n_points, track_trans=True)

        self.hidden_dim = hidden_dim
        self._obbox_embed = _obbox_embed = MLP(hidden_dim, hidden_dim, 5, 3)
        nn.init.constant_(_obbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_obbox_embed.layers[-1].bias.data, 0)
        box_embed_layerlist = [_obbox_embed for i in range(self.num_layers)]  # share box prediction each layer
        self.obbox_embed = nn.ModuleList(box_embed_layerlist)
        self.decoder.obbox_embed = self.obbox_embed
        self.with_position = with_position
        self.nms_query_select = nms_query_select
        self.temperature = temperature
        self.crop = crop
        self.new_pos = new_pos
        self.simi = simi
        self.dq = dq
        #判断是否为弱监督学习
        # if self.simi:
        #     self.levelset_bottom = nn.Conv2d(hidden_dim, 1, 3, padding=1)
        #判断是否采用dunamic_query
        if self.dq:
            self.ccm = CategoricalCounting(cls_num=3)  #(100, 300, 500)
            self.multiscale = MultiScaleFeature(is_5_scale=True)
            self.CGFE = CGFE(gate_channels=256, reduction_ratio=16, num_feature_levels=self.num_feature_levels)
            self.dynamic_query_list = [100, 300, 500]
            self.object_select = Objectselectattention(d_model=hidden_dim)
    @classmethod
    def from_config(cls, cfg, in_channels, mask_classification):
        ret = {}
        ret["in_channels"] = in_channels
        ret["mask_classification"] = mask_classification

        ret["num_classes"] = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        ret["hidden_dim"] = cfg.MODEL.MaskDINO.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MaskDINO.NUM_OBJECT_QUERIES
        # Transformer parameters:
        ret["nheads"] = cfg.MODEL.MaskDINO.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MaskDINO.DIM_FEEDFORWARD
        ret["dec_layers"] = cfg.MODEL.MaskDINO.DEC_LAYERS
        ret["enforce_input_project"] = cfg.MODEL.MaskDINO.ENFORCE_INPUT_PROJ
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["two_stage"] =cfg.MODEL.MaskDINO.TWO_STAGE
        ret["initialize_box_type"] = cfg.MODEL.MaskDINO.INITIALIZE_BOX_TYPE  # ['no', 'bitmask', 'mask2box']
        ret["dn"]=cfg.MODEL.MaskDINO.DN
        ret["noise_scale"] =cfg.MODEL.MaskDINO.DN_NOISE_SCALE
        ret["dn_num"] =cfg.MODEL.MaskDINO.DN_NUM
        ret["initial_pred"] =cfg.MODEL.MaskDINO.INITIAL_PRED
        ret["learn_tgt"] = cfg.MODEL.MaskDINO.LEARN_TGT
        ret["total_num_feature_levels"] = cfg.MODEL.SEM_SEG_HEAD.TOTAL_NUM_FEATURE_LEVELS
        ret["semantic_ce_loss"] = cfg.MODEL.MaskDINO.TEST.SEMANTIC_ON and cfg.MODEL.MaskDINO.SEMANTIC_CE_LOSS and ~cfg.MODEL.MaskDINO.TEST.PANOPTIC_ON
        ret["with_position"] = cfg.MODEL.SEM_SEG_HEAD.WITH_POSITION
        ret["nms_query_select"] = cfg.MODEL.SEM_SEG_HEAD.NMS_QUERY_SELECT
        ret["dn_aug"]  = cfg.MODEL.MaskDINO.DN_AUG
        ret["temperature"] = cfg.MODEL.TEMPERATURE
        ret["crop"] = cfg.CROP
        ret["new_pos"] = cfg.MODEL.SEM_SEG_HEAD.NEW_POS
        ret["simi"] = cfg.SIMI
        ret["dq"] = cfg.DQ
        return ret

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale

            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//(int(max(known_num)))
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_obbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_obbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_obbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_obbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_obboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_obbox_expand = known_obboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of obbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_obbox_expand)
                diff_xy = known_obbox_expand[:, 2:4] / 2 
                theta = known_obbox_expand[:, 4]
                diff[:, 0] = diff_xy[:, 0] * torch.cos(- theta * math.pi / 180) + diff_xy[:, 1] * torch.sin(- theta * math.pi / 180)
                diff[:, 1] = diff_xy[:, 1] * torch.cos(- theta * math.pi / 180) - diff_xy[:, 0] * torch.sin(- theta * math.pi / 180)
                diff[:, 2:4] = known_obbox_expand[:, 2:4]
                diff[:, 4] = torch.full_like(known_obbox_expand[:, 4], 45.0)
                known_obbox_expand += torch.mul((torch.rand_like(known_obbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_obbox_expand[:, 0:4] = known_obbox_expand[:, 0:4].clamp(min=0.0, max=1.0)
                known_obbox_expand[:, 4] = known_obbox_expand[:, 4].clamp(min=0.0, max=90.0) / 90.0
            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            input_obbox_embed = inverse_sigmoid(known_obbox_expand)
            single_pad = int(max(known_num))
            pad_size = int(single_pad * scalar)

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_obbox = torch.zeros(pad_size, 5).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_obbox = torch.cat([padding_obbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label=padding_label.repeat(batch_size, 1, 1)
                input_query_obbox = padding_obbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_obbox[(known_bid.long(), map_known_indice)] = input_obbox_embed

            tgt_size = pad_size + self.num_queries 
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_obboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_obbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_obbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_obbox is None:
            input_query_label = input_query_label
            input_query_obbox = input_query_obbox

        return input_query_label,input_query_obbox,attn_mask,mask_dict

    def prepare_for_dn_with_aug(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        # 加入重复框
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale
            fault_obbox_num = 30
            batch_size = len(targets)
            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k)*2 for k in known]
            single_pad = int(max(known_num)) 
            # use fix number of dn queries
            if max(known_num)>0:
                scalar = scalar//single_pad
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_obbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_obbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_obbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_obbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar * 2, 1).view(-1)
            known_bid = batch_idx.repeat(scalar * 2, 1).view(-1)
            known_obboxs = boxes.repeat(scalar * 2, 1)
            known_labels_expaned = known_labels.clone()
            known_obbox_expand = known_obboxs.clone()

            fault_obbox_label = torch.zeros(fault_obbox_num * batch_size, dtype=torch.int64).to(known_labels_expaned.device)
            known_labels_expaned = torch.cat((known_labels_expaned, fault_obbox_label), dim=0)
            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of obbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_obbox_expand)
                diff_xy = known_obbox_expand[:, 2:4] / 2 
                theta = known_obbox_expand[:, 4]
                diff[:, 0] = diff_xy[:, 0] * torch.cos(- theta * math.pi / 180) + diff_xy[:, 1] * torch.sin(- theta * math.pi / 180)
                diff[:, 1] = diff_xy[:, 1] * torch.cos(- theta * math.pi / 180) - diff_xy[:, 0] * torch.sin(- theta * math.pi / 180)
                diff[:, 2:4] = known_obbox_expand[:, 2:4]
                diff[:, 4] = torch.full_like(known_obbox_expand[:, 4], 45.0)
                known_obbox_expand += torch.mul((torch.rand_like(known_obbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_obbox_expand[:, 0:4] = known_obbox_expand[:, 0:4].clamp(min=0.0, max=1.0)
                known_obbox_expand[:, 4] = known_obbox_expand[:, 4].clamp(min=0.0, max=90.0) / 90.0

            #获取包含多个细胞的错误框
            arr = np.zeros((fault_obbox_num * batch_size, 2), dtype=np.int32)
            for i in range(fault_obbox_num):
                arr[i, 0] = np.random.randint(0, single_pad / 2)
                arr[i, 1] = np.random.randint(0, single_pad / 2)
                while arr[i, 1] == arr[i, 0]:
                    arr[i, 1] = np.random.randint(0, single_pad / 2)
            select_box = boxes[arr.flatten()]
            select_box = box_ops.obox_cxcywht_to_xyxy(select_box, norm_angle=False).reshape(fault_obbox_num * batch_size, 8, 1, 2)
            fault_obbox = []
            for i in select_box:
                r = cv2.minAreaRect(i.cpu().numpy())
                obbox = np.array([r[0][0], r[0][1], r[1][1], r[1][0], (90 - r[2])/90.0])  
                fault_obbox.append(obbox)
            fault_obbox = torch.tensor(fault_obbox).to(known_obbox_expand.device).type(torch.float32)
        

            m = known_labels_expaned.long().to('cuda')
            input_label_embed = self.label_enc(m)
            fault_obbox_label_embed = input_label_embed[- fault_obbox_num * batch_size:]
            input_label_embed = input_label_embed[:-fault_obbox_num * batch_size]
            input_obbox_embed = inverse_sigmoid(known_obbox_expand)
            pad_size = int(single_pad * scalar) 

            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_obbox = torch.zeros(pad_size, 5).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_obbox = torch.cat([padding_obbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                fault_obbox_label_embed = fault_obbox_label_embed.reshape(batch_size, fault_obbox_num, self.hidden_dim)
                fault_obbox = fault_obbox.reshape(batch_size, fault_obbox_num, 5)
                input_query_label=torch.cat((padding_label.repeat(batch_size, 1, 1), fault_obbox_label_embed),dim=1)
                input_query_obbox = torch.cat((padding_obbox.repeat(batch_size, 1, 1), fault_obbox),dim=1)
                
            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_obbox[(known_bid.long(), map_known_indice)] = input_obbox_embed


            tgt_size = pad_size + self.num_queries + fault_obbox_num
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size + fault_obbox_num:, :pad_size + fault_obbox_num] = True
            # TODO 让重建的框看不见query
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            # 错误的框看不见重建的框
            attn_mask[pad_size:pad_size+fault_obbox_num,:pad_size] = True

            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_obboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
                'fault_obbox': fault_obbox_num
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_obbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_obbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_obbox is None:
            input_query_label = input_query_label
            input_query_obbox = input_query_obbox

        return input_query_label,input_query_obbox,attn_mask,mask_dict


    def prepare_for_dn_with_tgt(self, targets, tgt, refpoint_emb, batch_size):
        """
        modified from dn-detr. You can refer to dn-detr 额外增加训练样本，即包含一个框包括多个样本的和重复检测的负样本。
        https://github.com/IDEA-Research/DN-DETR/blob/main/models/dn_dab_deformable_detr/dn_components.py
        for more details
            :param dn_args: scalar, noise_scale
            :param tgt: original tgt (content) in the matching part
            :param refpoint_emb: positional anchor queries in the matching part
            :param batch_size: bs
            """
        if self.training:
            scalar, noise_scale = self.dn_num,self.noise_scale
            
            known = [(torch.ones_like(t['labels'])).cuda() for t in targets]
            know_idx = [torch.nonzero(t) for t in known]
            known_num = [sum(k) for k in known]

            # use fix number of dn queries
            single_pad = int(max(known_num)) 
            if max(known_num)>0:
                scalar = scalar//(single_pad)
            else:
                scalar = 0
            if scalar == 0:
                input_query_label = None
                input_query_obbox = None
                attn_mask = None
                mask_dict = None
                return input_query_label, input_query_obbox, attn_mask, mask_dict

            # can be modified to selectively denosie some label or boxes; also known label prediction
            unmask_obbox = unmask_label = torch.cat(known)
            labels = torch.cat([t['labels'] for t in targets])
            boxes = torch.cat([t['boxes'] for t in targets])
            batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])
            # known
            known_indice = torch.nonzero(unmask_label + unmask_obbox)
            known_indice = known_indice.view(-1)

            # noise
            known_indice = known_indice.repeat(scalar, 1).view(-1)
            known_labels = labels.repeat(scalar, 1).view(-1)
            known_bid = batch_idx.repeat(scalar, 1).view(-1)
            known_obboxs = boxes.repeat(scalar, 1)
            known_labels_expaned = known_labels.clone()
            known_obbox_expand = known_obboxs.clone()

            # noise on the label
            if noise_scale > 0:
                p = torch.rand_like(known_labels_expaned.float())
                chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)  # half of obbox prob
                new_label = torch.randint_like(chosen_indice, 0, self.num_classes)  # randomly put a new one here
                known_labels_expaned.scatter_(0, chosen_indice, new_label)
            if noise_scale > 0:
                diff = torch.zeros_like(known_obbox_expand)
                diff_xy = known_obbox_expand[:, 2:4] / 2 
                theta = known_obbox_expand[:, 4]
                diff[:, 0] = diff_xy[:, 0] * torch.cos(- theta) + diff_xy[:, 1] * torch.sin(- theta)
                diff[:, 1] = diff_xy[:, 1] * torch.cos(- theta) - diff_xy[:, 0] * torch.sin(- theta)
                diff[:, 2:4] = known_obbox_expand[:, 2:4]
                diff[:, 4] = torch.full_like(known_obbox_expand[:, 4], 45.0)
                known_obbox_expand += torch.mul((torch.rand_like(known_obbox_expand) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                known_obbox_expand[:, 0:4] = known_obbox_expand[:, 0:4].clamp(min=0.0, max=1.0)
                known_obbox_expand[:, 4] = known_obbox_expand[:, 4].clamp(min=0.0, max=90.0) / 90.0


            #tgt = tgt.squeeze(0).permute(2,1,0) #(w,h,c)
            pos = known_obbox_expand[:, :2].clone()
            pos[:, 0] = pos[:, 0] * (tgt.shape[0] - 1)
            pos[:, 1] = pos[:, 1] * (tgt.shape[1] - 1)
            pos = pos.to(torch.long)
            input_label_embed = tgt[pos[:,0], pos[:,1]]
            input_obbox_embed = inverse_sigmoid(known_obbox_expand)
            pad_size = int(single_pad * scalar)
            padding_label = torch.zeros(pad_size, self.hidden_dim).cuda()
            padding_obbox = torch.zeros(pad_size, 5).cuda()

            if not refpoint_emb is None:
                input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
                input_query_obbox = torch.cat([padding_obbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
            else:
                input_query_label=padding_label.repeat(batch_size, 1, 1)
                input_query_obbox = padding_obbox.repeat(batch_size, 1, 1)

            # map
            map_known_indice = torch.tensor([]).to('cuda')
            if len(known_num):
                map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [1,2, 1,2,3]
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
            if len(known_bid):
                input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
                input_query_obbox[(known_bid.long(), map_known_indice)] = input_obbox_embed

            tgt_size = pad_size + self.num_queries 
            attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
            # match query cannot see the reconstruct
            attn_mask[pad_size:, :pad_size] = True
            # reconstruct cannot see each other
            for i in range(scalar):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == scalar - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                    
            mask_dict = {
                'known_indice': torch.as_tensor(known_indice).long(),
                'batch_idx': torch.as_tensor(batch_idx).long(),
                'map_known_indice': torch.as_tensor(map_known_indice).long(),
                'known_lbs_bboxes': (known_labels, known_obboxs),
                'know_idx': know_idx,
                'pad_size': pad_size,
                'scalar': scalar,
            }
        else:
            if not refpoint_emb is None:
                input_query_label = tgt.repeat(batch_size, 1, 1)
                input_query_obbox = refpoint_emb.repeat(batch_size, 1, 1)
            else:
                input_query_label=None
                input_query_obbox=None
            attn_mask = None
            mask_dict=None

        # 100*batch*256
        if not input_query_obbox is None:
            input_query_label = input_query_label
            input_query_obbox = input_query_obbox

        return input_query_label,input_query_obbox,attn_mask,mask_dict

    def dn_post_process(self,outputs_class,outputs_coord,mask_dict,outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1],'pred_masks': output_known_mask[-1]}

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask,output_known_coord)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_coord, outputs_mask

    def dn_post_process_aug(self,outputs_class,outputs_coord,mask_dict,outputs_mask):
        """
            post process of dn after output from the transformer
            put the dn part in the mask_dict
            """
        assert mask_dict['pad_size'] > 0
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'] + mask_dict['fault_obbox'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']+ mask_dict['fault_obbox']:, :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size']+ mask_dict['fault_obbox'], :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']+ mask_dict['fault_obbox']:, :]
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :mask_dict['pad_size']+ mask_dict['fault_obbox'], :]
            outputs_mask = outputs_mask[:, :, mask_dict['pad_size']+ mask_dict['fault_obbox']:, :]
        out = {'pred_logits': output_known_class[-1], 'pred_boxes': output_known_coord[-1],'pred_masks': output_known_mask[-1]}

        out['aux_outputs'] = self._set_aux_loss(output_known_class, output_known_mask,output_known_coord)
        mask_dict['output_known_lbs_bboxes']=out
        return outputs_class, outputs_coord, outputs_mask

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def pred_box(self, reference, hs, ref0=None):
        """
        :param reference: reference box coordinates from each decoder layer
        :param hs: content
        :param ref0: whether there are prediction from the first layer
        """
        device = reference[0].device
        if ref0 is None:
            outputs_coord_list = []
        else:
            outputs_coord_list = [ref0.to(device)]
        for dec_lid, (layer_ref_sig, layer_obbox_embed, layer_hs) in enumerate(zip(reference[:-1], self.obbox_embed, hs)):
            layer_delta_unsig = layer_obbox_embed(layer_hs).to(device)
            track_ref_box = layer_ref_sig[0, self.num_queries:]
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig).to(device)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            track_box = layer_outputs_unsig[0, self.num_queries:]
            outputs_coord_list.append(layer_outputs_unsig)
        outputs_coord_list = torch.stack(outputs_coord_list)
        return outputs_coord_list

    def forward(self, x, mask_features, masks, targets=None, tracker=None):
        """
        :param x: input, a list of multi-scale feature
        :param mask_features: is the per-pixel embeddings with resolution 1/4 of the original image,
        obtained by fusing backbone encoder encoded features. This is used to produce binary masks.
        :param masks: mask in the original image
        :param targets: used for denoising training 其中边框已经归一化
        :param trakcer: 细胞追踪轨迹
        """
        assert len(x) == self.num_feature_levels
        device = x[0].device
        size_list = []
        # disable mask, it does not affect performance
        enable_mask = 0
        if masks is not None:
            for src in x:
                if src.size(2) % 32 or src.size(3) % 32:
                    enable_mask = 1
        if enable_mask == 0:
            masks = [torch.zeros((src.size(0), src.size(2), src.size(3)), device=src.device, dtype=torch.bool) for src in x]
        src_flatten = []
        mask_flatten = []
        spatial_shapes = []

        split_size_or_sections = [None] * self.num_feature_levels
        for i in range(self.num_feature_levels):
            idx=self.num_feature_levels-1-i
            bs, c , h, w=x[idx].shape
            split_size_or_sections[i] = x[i].shape[-2] * x[i].shape[-1]
            size_list.append(x[i].shape[-2:])
            spatial_shapes.append(x[idx].shape[-2:])
            src_flatten.append(self.input_proj[idx](x[idx]).flatten(2).transpose(1, 2))
            mask_flatten.append(masks[i].flatten(1))
        src_flatten = torch.cat(src_flatten, 1)  # bs, \sum{hxw}, c
        mask_flatten = torch.cat(mask_flatten, 1)  # bs, \sum{hxw}
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1)
        #dynamic query
        if self.dq:
            counting_output, ccm_feature = self.ccm(src_flatten, spatial_shapes)
            multi_ccm_feature = self.multiscale(ccm_feature)
            cgfe_out = self.CGFE(multi_ccm_feature, src_flatten, spatial_shapes)
            memory = cgfe_out        
            _, predicted = torch.max(counting_output.data, 1)
            num_select = self.dynamic_query_list[max(predicted.tolist())]
            #num_select = 250
            tracker.pred_num = num_select
        predictions_class = []
        predictions_mask = []
        if self.two_stage:
            if self.dq:
                output_memory, output_proposals, vaild_mask = gen_encoder_output_proposals_ct(memory, mask_flatten, spatial_shapes)  #(x, y, w, h, theta)
            else:
                output_memory, output_proposals, vaild_mask = gen_encoder_output_proposals_ct(src_flatten, mask_flatten, spatial_shapes)  #(x, y, w, h, theta)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory).sigmoid()
            #将mask的特征点处的分类概率设为-10
            #enc_outputs_class_unselected[torch.where(~vaild_mask == True)] = -10
            enc_outputs_coord_unselected = self._obbox_embed(
                output_memory) + output_proposals  # (bs, \sum{hw}, 5) unsigmoid
            topk = self.topk
            if self.dq:
                if self.training:
                    topk = targets[0]["num_query"]
                    self.num_queries = topk
                else:
                    topk = num_select
                    self.num_queries = topk
            if self.nms_query_select and not self.training:
                topk_proposals = torch.topk(enc_outputs_class_unselected[:, :, 0], 300, dim=1)[1] #选取400个topk个值
            
                scores = torch.gather(enc_outputs_class_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 2)) 
            else:
                topk_proposals = torch.topk(enc_outputs_class_unselected[:, :, 0], topk, dim=1)[1] #只需要cell概率最大的topk个值
            refpoint_embed_undetach = torch.gather(enc_outputs_coord_unselected, 1,
                                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 5))  # unsigmoid
            #假设batch_size = 1, 加入NMS过滤 似乎是负作用，会过滤掉一些检测,可以降低NMS阈值
            if self.nms_query_select and not self.training:
                _, _, _, index = box_ops.multiclass_nms_rotated(refpoint_embed_undetach.squeeze(0).sigmoid(), scores.squeeze(0), -10, 0.5, topk, return_inds=True)
                last_num = topk - index.shape[0]
                if last_num > 0:
                    query_num = torch.arange(topk, device=index.device)
                    last_index = torch.tensor([x for x in query_num if x not in index], device=index.device)
                    index = torch.cat((index, last_index[:last_num]))
                refpoint_embed_undetach = refpoint_embed_undetach[0, index].unsqueeze(0)
                topk_proposals = topk_proposals[0, index].unsqueeze(0)
                
            refpoint_embed = refpoint_embed_undetach.detach()

            tgt_undetach = torch.gather(output_memory, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid
            if self.with_position:
                if not self.crop:
                    outputs_class, outputs_mask = self.forward_prediction_heads_with_position(tgt_undetach.transpose(0, 1), mask_features, refpoint_embed_undetach.sigmoid().transpose(0, 1), valid_ratios, item=True)
                else: 
                    outputs_class, outputs_mask = self.forward_prediction_heads_with_position_crop(tgt_undetach.transpose(0, 1), mask_features, refpoint_embed_undetach.sigmoid().transpose(0, 1), valid_ratios)
            else:
                outputs_class, outputs_mask = self.forward_prediction_heads(tgt_undetach.transpose(0, 1), mask_features, item=True)
            tgt = tgt_undetach.detach()
            if self.learn_tgt:
                tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            interm_outputs=dict()
            interm_outputs['query'] = tgt
            interm_outputs['pred_logits'] = outputs_class
            interm_outputs['pred_boxes'] = refpoint_embed_undetach.sigmoid()
            interm_outputs['pred_masks'] = outputs_mask

            if self.initialize_box_type != 'no':  #尝试不在训练时加入bitmask 无用
                # convert masks into boxes to better initialize box in the decoder
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                if self.initialize_box_type == 'bitmask':  # slower, but more accurate
                    if not self.crop:
                        refpoint_embed = BitMasks_ct(flaten_mask > 0).get_oriented_bounding_boxes().to(device)  #对斜边框角度进行归一化
                        refpoint_embed = box_ops.scale_obbox(refpoint_embed, torch.tensor([1/w, 1/h], dtype=float, device=refpoint_embed.device), norm_angle= True)
                    else:
                        refpoint_embed_crop = BitMasks_ct(flaten_mask > 0).get_oriented_bounding_boxes().to(device)  #对斜边框角度进行归一化
                        #refpoint_embed_crop = box_ops.scale_obbox(refpoint_embed_crop, torch.tensor([1/w, 1/h], dtype=float, device=device), norm_angle= True)
                        H, W = mask_features.shape[-2:]
                        refpoint_embed_crop[..., :2] = refpoint_embed_crop[..., :2] + refpoint_embed[..., :2].sigmoid() * torch.tensor([W, H], device=device) - torch.tensor([w/2, h/2], device=device)
                        refpoint_embed = box_ops.scale_obbox(refpoint_embed_crop, torch.tensor([1/W, 1/H], dtype=float, device=refpoint_embed.device), norm_angle= True)
                elif self.initialize_box_type == 'mask2box':  # faster conversion
                    refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
                else:
                    assert NotImplementedError
                #refpoint_embed = refpoint_embed / torch.as_tensor([w, h, w, h, 90], dtype=torch.float).to(device)  #对边框进行归一化
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0], outputs_mask.shape[1], 5) #(bs, num_query, 5)
                object_query_pos = refpoint_embed.clone() #用于矫正轨迹位置
                refpoint_embed = inverse_sigmoid(refpoint_embed) #后面要进行sigmoid
        elif not self.two_stage:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)

        tgt_mask = None
        mask_dict = None
        dn_num = 0
        if self.dn != "no" and self.training:
            assert targets is not None
            if self.dn_aug:
                input_query_label, input_query_obbox, tgt_mask, mask_dict = \
                self.prepare_for_dn_with_aug(targets, None, None, x[0].shape[0])     #对角度进行归一化
            else:
                input_query_label, input_query_obbox, tgt_mask, mask_dict = \
                self.prepare_for_dn(targets, None, None, x[0].shape[0])     #对角度进行归一化
            # 采用特征图中相应位置的特征作为query
            # y = torch.split(output_memory, split_size_or_sections, dim=1)
            # out = []
            # for i, z in enumerate(y):
            #     out.append(z.transpose(1, 2).view(bs, -1, size_list[i][0], size_list[i][1]))
            # input_query_label, input_query_obbox, tgt_mask, mask_dict = self.prepare_for_dn_with_tgt(targets, out[-1].squeeze().permute(2,1,0), None, x[0].shape[0])
            if mask_dict is not None:
                tgt=torch.cat([input_query_label, tgt],dim=1)
                refpoint_embed=torch.cat([input_query_obbox,refpoint_embed],dim=1)
            dn_num = input_query_obbox.shape[1]

        # 在训练时将追踪轨迹加入整个query中,应对有丝分裂，将track query进行复制
        track_query=None
        track_pos=None
        if  tracker.track_num > 0:
            if self.training:
                N = refpoint_embed.shape[1]
                n = tracker.max_num
                if not self.simi or tracker.reverse:
                    #随机抛弃一些query
                    # 1. 随机生成丢弃比例（0% 到 10%）
                    drop_ratio = random.uniform(0, 0.2)
                    # 2. 计算需要保留的样本数量
                    k = max(1, int(n * (1 - drop_ratio)))
                    # 3. 随机选择保留的样本索引
                    indices = torch.randperm(n)[:k]
                    tracker.track_query = tracker.track_query[:, indices]
                    tracker.track_pos = tracker.track_pos[:, indices]
                    for i in range(len(tracker.track_ids)):
                        tracker.track_ids[i] = tracker.track_ids[i][indices]
                        targets[i]["track_id"] = targets[i]["track_id"][indices]
                    tracker.max_num = k
                track_query = tracker.track_query.transpose(0, 1)
                track_pos = tracker.track_pos.transpose(0, 1)
                #训练是在track_pos中加入一定的噪声
                noise_scale=0.1
                diff = torch.zeros_like(track_pos)
                diff_xy = track_pos[:, :, 2:4] / 2 
                theta = track_pos[:,:, 4] * math.pi / 2 
                diff[:, :, 0] = diff_xy[:, :, 0] * torch.cos(- theta) + diff_xy[:, :, 1] * torch.sin(- theta)
                diff[:, :, 1] = diff_xy[:, :, 1] * torch.cos(- theta) - diff_xy[:, :, 0] * torch.sin(- theta)
                diff[:, :, 2:4] = track_pos[:, :, 2:4]
                diff[:, :, 4] = torch.full_like(track_pos[:, :, 4], 0.5)
                track_pos += torch.mul((torch.rand_like(track_pos) * 2 - 1.0),
                                               diff).cuda() * noise_scale
                track_pos[:, :, 0:5] = track_pos[:, :, 0:5].clamp(min=0.0, max=1.0)
                #track_pos = torch.cat([track_pos, track_pos_noise], dim=0)  #进行复制           
                #调整track_query
                reference_points_input = box_ops.scale_obbox((track_pos)[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], self.temperature) # nq, bs, 128 * 5
                if self.new_pos:
                    query_pos = query_sine_embed[:,:,0:256] + self.decoder.ref_point_head(query_sine_embed[:,:,256:])
                else:
                    query_pos = self.decoder.ref_point_head(query_sine_embed)  # nq, bs, 256
                track_query = self.track_query_process(tgt=track_query,
                                                       tgt_query_pos=query_pos,
                                                       tgt_reference_points=reference_points_input,
                                                       memory=src_flatten.transpose(0, 1),
                                                       memory_key_padding_mask=mask_flatten,
                                                       memory_level_start_index=level_start_index,
                                                       memory_spatial_shapes=spatial_shapes,)    
                reference_before_sigmoid = inverse_sigmoid(track_pos)
                delta_unsig = self._obbox_embed(track_query).to(device)
                track_pos = delta_unsig + reference_before_sigmoid
                track_pos = track_pos.transpose(0, 1)
                track_query = track_query.transpose(0, 1)

                #对齐后的track query进行监督训练
                # outputs_class_track, outputs_mask_track = self.forward_prediction_heads_with_position(track_query, mask_features, track_pos.sigmoid(), valid_ratios)
                # track_outputs=dict()
                # track_outputs['pred_logits'] = outputs_class_track
                # track_outputs['pred_boxes'] = track_pos.transpose(0,1).sigmoid()
                # track_outputs['pred_masks'] = outputs_mask_track
                # track_pos = track_pos.transpose(0, 1).detach()
                # track_query = track_query.transpose(0, 1).detach()

                #调整object_query
                # tgt = tgt.transpose(0,1)
                # refpoint_embed = refpoint_embed.transpose(0,1)
                # reference_points_input = box_ops.scale_obbox((refpoint_embed.sigmoid())[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
                # query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], self.temperature) # nq, bs, 128 * 5
                # if self.new_pos:
                #     query_pos = query_sine_embed[:,:,0:256] + self.decoder.ref_point_head(query_sine_embed[:,:,256:])
                # else:
                #     query_pos = self.decoder.ref_point_head(query_sine_embed)  # nq, bs, 256
                # tgt = self.detection_query_process( tgt=tgt,
                #                                 tgt_query_pos=query_pos,
                #                                 tgt_reference_points=reference_points_input,
                #                                 memory=src_flatten.transpose(0, 1),
                #                                 memory_key_padding_mask=mask_flatten,
                #                                 memory_level_start_index=level_start_index,
                #                                 memory_spatial_shapes=spatial_shapes,)    
                # delta_unsig = self._obbox_embed(tgt).to(device)
                # refpoint_embed = delta_unsig + refpoint_embed
                # refpoint_embed = refpoint_embed.transpose(0, 1)
                # tgt = tgt.transpose(0, 1)
                #加入第二阶段动态查询
                if self.dq:
                    track_ids = targets[0]["track_id"]
                    target_ids = targets[0]["gt_id"]
                    #提取相邻两帧中相同对应id
                    combined = torch.cat((track_ids,target_ids), dim=0)
                    combined_val, counts = combined.unique(return_counts=True)
                    common_elements = combined_val[counts>1]
                    num_new_object= len(target_ids) - len(common_elements)
                    tgt_dq = torch.cat([tgt, track_query],dim=1)
                    refpoint_embed_dq = torch.cat([refpoint_embed, track_pos],dim=1)
                    reference_points_input = box_ops.scale_obbox((refpoint_embed_dq.sigmoid())[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], temperature=self.temperature) # nq, bs, 128 * 5
                    if self.new_pos:
                        query_pos =  query_sine_embed[:,:,0:256] + self.decoder.ref_point_head(query_sine_embed[:,:,256:])
                        query_pos = query_pos.transpose(0,1)
                    else:
                        query_pos = self.decoder.ref_point_head(query_sine_embed).transpose(0,1)  # bs, nq, 256
                    topk_proposals, object_cls_scores , cls_scores = self.object_select(tgt_dq.transpose(0, 1) ,query_pos, num_object_query=self.num_queries ,training=self.training, num_new_object=num_new_object)
                    tgt = torch.gather(tgt, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
                    refpoint_embed = torch.gather(refpoint_embed, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, 5)).sigmoid()
                    outputs_mask = torch.gather(outputs_mask, 1,
                                  topk_proposals.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, outputs_mask.shape[-2], outputs_mask.shape[-1]))
                    # if topk_proposals.shape[1] != self.num_queries:
                    #     print("yes")
                    self.num_queries = topk_proposals.shape[1]
                    dq_outputs=dict()
                    dq_outputs['pred_logits'] = object_cls_scores
                    dq_outputs['pred_boxes'] = refpoint_embed.sigmoid()
                    dq_outputs['pred_masks'] = outputs_mask
                    dq_outputs['pred_logits_all'] = cls_scores
                    targets[0]["num_query"] = self.num_queries
                tgt = torch.cat([tgt, track_query],dim=1)
                refpoint_embed = torch.cat([refpoint_embed, track_pos],dim=1)
                M = refpoint_embed.shape[1]
                if self.dn!="no":
                    padding_tgt_mask = torch.full((M, M), False, device=tgt_mask.device)
                    padding_tgt_mask[:N, :N] = tgt_mask
                    padding_tgt_mask[N:M, :dn_num] = True
                    tgt_mask = padding_tgt_mask
                    del padding_tgt_mask
                    del input_query_obbox
                    del input_query_label
            else:
                track_query = tracker.track_query.transpose(0, 1)
                track_pos = tracker.track_pos
                # track_motion_cls  = self.motion_cls_pred(track_query.transpose(0, 1)).sigmoid()
                # print(track_motion_cls.squeeze())
                # track_pos = inverse_sigmoid(track_pos)
                # 运动预测
                object_query_score = interm_outputs['pred_logits'].sigmoid()
                #object_query_center_pos = object_query_pos[:, :, :2]
                #过滤掉成绩过低的检测
                object_query_score = object_query_score[:, :, 0]
                object_query_pos = object_query_pos[object_query_score >= (tracker.track_obj_score_thresh)].unsqueeze(0)
                object_query_score = object_query_score[object_query_score >= (tracker.track_obj_score_thresh)].unsqueeze(0)
                #采用中心点距离过滤掉重复的检测
                indices_last = nms_distance(object_query_pos[0, : , :2], object_query_score[0], 0.01)
                indices_last = torch.tensor(indices_last).unsqueeze(0).unsqueeze(-1).expand(-1, -1, object_query_pos.size(-1)).to(object_query_pos.device)
                object_query_pos = torch.gather(object_query_pos, 1, indices_last)
                #主要针对Fluo-N2DH-GOWT1数据集,计算检测面积
                object_area = object_query_pos[0, :, 2] * object_query_pos[0, :, 3]
                object_area = object_area.unsqueeze(0).repeat(tracker.track_num, 1)
                # if tracker.frame_index == 4:
                #     print("ok")
                #     print(track_pos)
                #     print(object_query_pos)
                #     print(tracker.track_ids)
                # object_query_center_pos[object_query_center_pos > 0.2] = 0
                # 扩展维度以便能够进行广播操作
                expanded_tensor1 = track_pos.unsqueeze(2)  # shape: [1, N, 1, 2]
                expanded_tensor2 = object_query_pos.unsqueeze(1)  # shape: [1, 1, M, 2]
                # 计算欧氏距离  
                diff = expanded_tensor2 - expanded_tensor1  # 计算差值
                distance = torch.sqrt(torch.sum(diff[:, :, :, :2] ** 2, dim=-1))  # 计算欧氏距离
                #每个检测只保留一个和追踪之间的最小距离
                # 找到每一列的最小值
                min_values, _ = torch.min(distance, dim=1)
                # 将除最小值外的其他值设为1
                distance = torch.where(distance == min_values.unsqueeze(1), distance, 1.0)
                # 获取每个 tensor1 中值对应的最小距离索引（最小的两个）
                min_diatance, min_indices = torch.topk(distance, k=2, dim=2, largest=False)  # 获取最小距离索引
                # 根据最小距离索引获取对应的 tensor2 中的值
                min_diff = torch.gather(diff, 2, min_indices.unsqueeze(-1).repeat(1, 1, 1, 5))
                # 面积
                min_area = torch.gather(object_area, 1, min_indices[0])
                mask_area = torch.all(min_area< tracker.min_area, dim=1)
                indices_area = torch.nonzero(mask_area)
                min_diff_copy = min_diff.clone()
                if tracker.frame_index == 46 and tracker.dataset=="DIC-C2DH-HeLa":
                    min_diff[min_diatance > 0.15] = 0
                else:
                    min_diff[min_diatance > tracker.track_min_distance] = 0
                if tracker.frame_index in tracker.special_frame:
                    min_diff[min_diatance > 0] = 0
                # 创建一个索引掩码，用于找到满足条件的元素
                mask = torch.all(min_diff[:, :, 1, :] == torch.tensor([0, 0, 0, 0, 0], dtype=torch.float64, device=min_diff.device), dim=2)
                # 获取满足条件的元素的索引
                indices = torch.nonzero(mask)
                # 使用索引替换满足条件的元素
                if tracker.dataset=="Fluo-N2DH-SIM+":
                    min_diff[indices[:, 0], indices[:, 1], 1] = min_diff[indices[:, 0], indices[:, 1], 0] = 0
                else:
                    min_diff[indices[:, 0], indices[:, 1], 1] = min_diff[indices[:, 0], indices[:, 1], 0]
                # 使用面积去判断分裂对于Fluo-N2DH-GOWT1
                if len(indices_area) > 0 and tracker.dataset=="Fluo-N2DH-GOWT1":
                    min_diff[0, indices_area[:,0]] = min_diff_copy[0, indices_area[:, 0]]
                #min_diff[indices[:, 0], indices[:, 1]] = min_diff[indices[:, 0], indices[:, 1]] * 0
                min_diff = torch.cat([min_diff[:, :, 0, :], min_diff[:, :, 1, :]], dim=1)
    
                track_pos = torch.cat((track_pos, track_pos), dim=1)
                track_pos[:, :, :] = track_pos[:, :, :] + min_diff
                # 将不可能发生分裂的置为全0
                # track_pos = torch.gather(object_query_pos, 2, min_indices.unsqueeze(-1).repeat(1, 1, 1, 5))
                track_pos = track_pos.transpose(0, 1)
                #将query和pos进行复制
                track_query = torch.cat([track_query, track_query], dim=0)
                #调整track_qeury
                reference_points_input = box_ops.scale_obbox((track_pos)[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
                query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], self.temperature) # nq, bs, 128 * 5
                if self.new_pos:
                    query_pos = query_sine_embed[:,:,0:256] + self.decoder.ref_point_head(query_sine_embed[:,:,256:])
                else:
                    query_pos = self.decoder.ref_point_head(query_sine_embed)  # nq, bs, 256
                track_query = self.track_query_process(tgt=track_query,
                                                       tgt_query_pos=query_pos,
                                                       tgt_reference_points=reference_points_input,
                                                       memory=src_flatten.transpose(0, 1),
                                                       memory_key_padding_mask=mask_flatten,
                                                       memory_level_start_index=level_start_index,
                                                       memory_spatial_shapes=spatial_shapes,)
                reference_before_sigmoid = inverse_sigmoid(track_pos)
                delta_unsig = self._obbox_embed(track_query).to(device)
                track_pos = delta_unsig + reference_before_sigmoid
                track_pos = track_pos.transpose(0, 1)
                track_query = track_query.transpose(0, 1)
                #调整object_query
                # tgt = tgt.transpose(0,1)
                # refpoint_embed = refpoint_embed.transpose(0,1)
                # reference_points_input = box_ops.scale_obbox((refpoint_embed.sigmoid())[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
                # query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], self.temperature) # nq, bs, 128 * 5
                # if self.new_pos:
                #     query_pos = query_sine_embed[:,:,0:256] + self.decoder.ref_point_head(query_sine_embed[:,:,256:])
                # else:
                #     query_pos = self.decoder.ref_point_head(query_sine_embed)  # nq, bs, 256
                # tgt = self.detection_query_process( tgt=tgt,
                #                                 tgt_query_pos=query_pos,
                #                                 tgt_reference_points=reference_points_input,
                #                                 memory=src_flatten.transpose(0, 1),
                #                                 memory_key_padding_mask=mask_flatten,
                #                                 memory_level_start_index=level_start_index,
                #                                 memory_spatial_shapes=spatial_shapes,)    
                # delta_unsig = self._obbox_embed(tgt).to(device)
                # refpoint_embed = delta_unsig + refpoint_embed
                # refpoint_embed = refpoint_embed.transpose(0, 1)
                # tgt = tgt.transpose(0, 1)
                #加入第二阶段动态查询
                if self.dq:
                    tgt_dq = torch.cat([tgt, track_query],dim=1)
                    refpoint_embed_dq = torch.cat([refpoint_embed, track_pos],dim=1)
                    reference_points_input = box_ops.scale_obbox((refpoint_embed_dq.sigmoid())[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
                    query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], temperature=self.temperature) # nq, bs, 128 * 5
                    if self.new_pos:
                        query_pos =  query_sine_embed[:,:,0:256] + self.decoder.ref_point_head(query_sine_embed[:,:,256:])
                        query_pos = query_pos.transpose(0,1)
                    else:
                        query_pos = self.decoder.ref_point_head(query_sine_embed).transpose(0,1)  # bs, nq, 256
                    topk_proposals = self.object_select(tgt_dq.transpose(0, 1) ,query_pos, num_object_query=self.num_queries ,training=self.training)
                    tgt = torch.gather(tgt, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
                    refpoint_embed = torch.gather(refpoint_embed, 1,
                                  topk_proposals.unsqueeze(-1).repeat(1, 1, 5)).sigmoid()
                    outputs_mask = torch.gather(outputs_mask, 1,
                                  topk_proposals.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, outputs_mask.shape[-2], outputs_mask.shape[-1]))
                    self.num_queries = topk_proposals.shape[1]
                    num_select = self.num_queries

                tgt = torch.cat([tgt, track_query],dim=1)
                refpoint_embed = torch.cat([refpoint_embed, track_pos],dim=1)
    
        # direct prediction from the matching and denoising part in the begining
        if self.initial_pred and self.training:
            if self.with_position:
                outputs_class, outputs_mask = self.forward_prediction_heads_with_position(tgt.transpose(0, 1), mask_features, refpoint_embed.sigmoid().transpose(0,1), valid_ratios, self.training)
            else:
                outputs_class, outputs_mask = self.forward_prediction_heads(tgt.transpose(0, 1), mask_features, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        #将深度特征加入targets中
        # if self.simi and self.training:
        #     mask_features_simi = mask_features.clone().detach()
        #     fea_images = self.levelset_bottom(mask_features_simi)
        #     if  tracker.track_num > 0:
        #         fea_images = fea_images.detach()
        #     for target, fea_image in zip(targets, fea_images):
        #         target["fea_image"] = fea_image.squeeze(0)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            tgt_track = track_query,
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=None,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            refpoints_unsigmoid_track=track_pos,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask
        )
        # iteratively box prediction
        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            #assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)

        for i, output in enumerate(hs):
            if self.with_position:
                if not self.crop:
                    outputs_class, outputs_mask = self.forward_prediction_heads_with_position(output.transpose(0, 1), mask_features, out_boxes[i +1].transpose(0,1), valid_ratios, self.training or (i == len(hs)-1))
                else:
                    outputs_class, outputs_mask = self.forward_prediction_heads_with_position_crop(output.transpose(0, 1), mask_features, out_boxes[i +1].transpose(0,1), valid_ratios, self.training or (i == len(hs)-1))
            else:
                outputs_class, outputs_mask = self.forward_prediction_heads(output.transpose(0, 1), mask_features, self.training or (i == len(hs)-1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if mask_dict is not None:
            predictions_mask=torch.stack(predictions_mask)
            predictions_class=torch.stack(predictions_class)
            if self.dn_aug:
                predictions_class, out_boxes,predictions_mask=\
                    self.dn_post_process_aug(predictions_class,out_boxes,mask_dict,predictions_mask)
            else:
                predictions_class, out_boxes,predictions_mask=\
                    self.dn_post_process(predictions_class,out_boxes,mask_dict,predictions_mask)
            predictions_class,predictions_mask=list(predictions_class),list(predictions_mask)
        elif self.training:  # this is to insure self.label_enc participate in the model
            predictions_class[-1] += 0.0*self.label_enc.weight.sum()

        out = {
            'pred_logits': predictions_class[-1],
            'pred_masks': predictions_mask[-1],
            'pred_boxes':out_boxes[-1],
            'query': hs[-1][:, dn_num:, :],
            # 'pred_motion_cls': track_motion_cls, 
            # 'pred_motion': track_pos_undetach,
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_mask,out_boxes
            )
        }
        if self.two_stage:
            out['interm_outputs'] = interm_outputs
        # if tracker.track_num and self.training > 0:
        #     out['track_outputs'] = track_outputs
        if self.dq:
            out['num_select'] = num_select
            if self.training:
                out['counting_output'] = counting_output
                if tracker.track_num > 0:
                    out['dq_outputs'] = dq_outputs
        return out, mask_dict


    def forward_prediction_heads(self, output, mask_features, pred_mask=True, item=False):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output).float()
        outputs_mask = None
        if pred_mask:
            if item:
                mask_embed = self.mask_embed_item(decoder_output)
            else:
                mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        return outputs_class, outputs_mask

    def forward_prediction_heads_with_position(self, output, mask_features, reference_points, valid_ratios, pred_mask=True, item=False):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output).float()
        outputs_mask = None
        if pred_mask:
            reference_points_input = box_ops.scale_obbox((reference_points)[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], temperature=self.temperature) # nq, bs, 128 * 5
            if self.new_pos:
                query_pos =  query_sine_embed[:,:,0:256] + self.decoder.ref_point_head(query_sine_embed[:,:,256:])
                query_pos = query_pos.transpose(0,1)
            else:
                query_pos = self.decoder.ref_point_head(query_sine_embed).transpose(0,1)  # bs, nq, 256
            if item:
                mask_embed = self.mask_embed_item(decoder_output + query_pos)
            else:
                mask_embed = self.mask_embed(decoder_output + query_pos)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
            # 测试位置注意力机制，简单采用MLP映射并不能将位置编码映射为想要的样子
            # pe_layer = PositionEmbeddingSine(128, temperature=self.temperature, normalize=True)
            # test_pos = torch.tensor([[[0, 0, 0.1, 0.5, 0]]], device="cuda")
            # test_pos_embed = gen_sineembed_for_position(test_pos, temperature=self.temperature)
            # if self.new_pos:
            #     test_pos_embed =  test_pos_embed[:,:,0:256] + self.decoder.ref_point_head(test_pos_embed[:,:,256:])
            #     test_pos_embed = test_pos_embed.transpose(0,1)
            # else:
            #     test_pos_embed = self.decoder.ref_point_head(test_pos_embed).transpose(0,1)
            # map = torch.ones([1, 1, 256, 256],device="cuda")
            # map_pos = pe_layer(map) #bchw
            # r = torch.einsum("bqc,bchw->bqhw",test_pos_embed, map_pos)
            # r = r.squeeze()
            # colors = [(1, 1, 1), (25/255,25/255,112/255)]  # 从白色到深蓝色
            # cmap = LinearSegmentedColormap.from_list('custom_cmap', colors)
            # #cmap = ListedColormap(colors)
            # plt.imshow(r.cpu(), cmap=cmap, interpolation='nearest')
            # plt.colorbar()  # 添加颜色条
            # plt.show()

        return outputs_class, outputs_mask

    def forward_prediction_heads_with_position_crop(self, output, mask_features, reference_points, valid_ratios, pred_mask=True):
        decoder_output = self.decoder_norm(output)
        decoder_output = decoder_output.transpose(0, 1)
        outputs_class = self.class_embed(decoder_output).float()
        outputs_mask = None
        if pred_mask:
            region_max_width, region_max_height = get_rotated_rect_vertices(reference_points)  #(B)
            region_features = extract_region_features(mask_features, reference_points[:,:,:2], region_max_width, region_max_height)
            reference_points_input = box_ops.scale_obbox((reference_points)[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], temperature=self.temperature) # nq, bs, 128 * 5
            if self.new_pos:
                query_pos =  query_sine_embed[:,:,0:256] + self.decoder.ref_point_head(query_sine_embed[:,:,256:])
                query_pos = query_pos.transpose(0,1)
            else:
                query_pos = self.decoder.ref_point_head(query_sine_embed).transpose(0,1)  # bs, nq, 256
            mask_embed = self.mask_embed(decoder_output + query_pos)
            outputs_mask = torch.einsum("bqc,bqchw->bqhw", mask_embed, region_features)
        return outputs_class, outputs_mask

    # def multi_forward_prediction_heads(self, output, mask_features, pred_mask=True):
    #     weights = self.weight_feature(output)
    #     weights = self.softmax(weights).transpose(0, 1).unsqueeze(dim=2)
    #     decoder_output = self.decoder_norm(output)
    #     decoder_output = decoder_output.transpose(0, 1)
    #     outputs_class = self.class_embed(decoder_output)
    #     outputs_mask = None
    #     if pred_mask:
    #         mask_embed = self.mask_embed(decoder_output)
    #         weight = weights[..., -1].unsqueeze(-1)
    #         outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features) * weight
    #         size = outputs_mask.shape[-2:]
    #         for i, mask_feature in enumerate(mask_features[0:-1]):
    #             weight =weights[..., i].unsqueeze(-1)
    #             mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_feature)
    #             mask = F.interpolate(mask, size=size, mode="bilinear", align_corners=False)
    #             outputs_mask = outputs_mask + weight * mask
        
    #     return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        # if self.mask_classification:
        if out_boxes is None:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_boxes":c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])
            ]



class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False,
                 d_model=256, query_dim=5,
                 modulate_hw_attn=True,
                 num_feature_levels=1,
                 deformable_decoder=True,
                 decoder_query_perturber=None,
                 dec_layer_number=None,  # number of queries each layer in decoder
                 rm_dec_query_scale=True,
                 dec_layer_share=False,
                 dec_layer_dropout_prob=None,
                 temperature=100000,
                 new_pos=False,
                 ):
        super().__init__()
        if num_layers > 0:
            self.layers = _get_clones(decoder_layer, num_layers, layer_share=dec_layer_share)
        else:
            self.layers = []
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate
        assert return_intermediate, "support return_intermediate only"
        self.query_dim = query_dim
        assert query_dim in [2, 4, 5], "query_dim should be 2/4/5 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.new_pos = new_pos
        if self.new_pos:
            self.ref_point_head = MLP(128 * 3, d_model, d_model, 4)
        else:
            self.ref_point_head = MLP(128 * 5, d_model, d_model, 2)
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.obbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers
            # assert dec_layer_number[0] ==

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0
        self.temperature = temperature
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()

    def forward(self, tgt, memory,
                tgt_track = None,
                refpoints_unsigmoid_track = None,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                refpoints_unsigmoid: Optional[Tensor] = None,  # num_queries, bs, 2
                # for memory
                level_start_index: Optional[Tensor] = None,  # num_levels
                spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                valid_ratios: Optional[Tensor] = None,

                ):
        """
        Input:
            - tgt: nq, bs, d_model
            - memory: hw, bs, d_model
            - pos: hw, bs, d_model
            - refpoints_unsigmoid: nq, bs, 2/4/5
            - valid_ratios/spatial_shapes: bs, nlevel, 2
        """
        output = tgt
        device = tgt.device

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid().to(device)
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):
            # if layer_id==1 and tgt_track!=None:
            #         output = torch.cat([output, tgt_track], dim=0)
            #         ref_points[-1] = torch.cat([ref_points[-1], refpoints_unsigmoid_track.sigmoid()], dim=0)
            #         reference_points = torch.cat([reference_points, refpoints_unsigmoid_track.sigmoid()], dim=0)
            # preprocess ref points
            if self.training and self.decoder_query_perturber is not None and layer_id != 0:
                reference_points = self.decoder_query_perturber(reference_points)

            reference_points_input = box_ops.scale_obbox((reference_points)[:, :, None], valid_ratios[None, :], norm_angle= True).to(torch.float32)  # nq, bs, nlevel, 5                                       
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], self.temperature) # nq, bs, 128 * 5
            if self.new_pos:
                raw_query_pos = query_sine_embed[:,:,0:256] + self.ref_point_head(query_sine_embed[:,:,256:])
            else:
                raw_query_pos = self.ref_point_head(query_sine_embed)  # nq, bs, 256
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            
            query_pos = pos_scale * raw_query_pos

            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_query_sine_embed=query_sine_embed,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_reference_points=reference_points_input,

                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                memory_pos=pos,

                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask
            )

            # iter update
            if self.obbox_embed is not None:
                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.obbox_embed[layer_id](output).to(device)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()
                reference_points = new_reference_points.detach()
                # if layer_id != self.num_layers - 1:
                ref_points.append(new_reference_points)

            intermediate.append(self.norm(output))

        return [
            [itm_out.transpose(0, 1) for itm_out in intermediate],
            [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points]
        ]


class DeformableTransformerDecoderLayer(nn.Module):

    def __init__(self, d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 use_deformable_box_attn=False,
                 key_aware_type=None,
                 track_trans=False,
                 ):
        super().__init__()

        # cross attention
        if use_deformable_box_attn:
            raise NotImplementedError
        else:
            self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points, motion_pred=False)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        if track_trans:
            self.self_attn = None
        self.motion_pred = False
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout3 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout)
        self.norm3 = nn.LayerNorm(d_model)

        self.key_aware_type = key_aware_type
        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    @autocast(enabled=False)
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                tgt_query_sine_embed: Optional[Tensor] = None,  # pos for query. Sine(pos)
                tgt_key_padding_mask: Optional[Tensor] = None,
                tgt_reference_points: Optional[Tensor] = None,  # nq, bs, 4

                # for memory
                memory: Optional[Tensor] = None,  # hw, bs, d_model
                memory_key_padding_mask: Optional[Tensor] = None,
                memory_level_start_index: Optional[Tensor] = None,  # num_levels
                memory_spatial_shapes: Optional[Tensor] = None,  # bs, num_levels, 2
                memory_pos: Optional[Tensor] = None,  # pos for memory

                # sa
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                cross_attn_mask: Optional[Tensor] = None,  # mask used for cross-attention
                ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # cross attention
        if self.key_aware_type is not None:
            if self.key_aware_type == 'mean':
                tgt = tgt + memory.mean(0, keepdim=True)
            elif self.key_aware_type == 'proj_mean':
                tgt = tgt + self.key_aware_proj(memory).mean(0, keepdim=True)
            else:
                raise NotImplementedError("Unknown key_aware_type: {}".format(self.key_aware_type))
        tgt2 = self.cross_attn(self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
                               tgt_reference_points.transpose(0, 1).contiguous(),
                               memory.transpose(0, 1), memory_spatial_shapes, memory_level_start_index,
                               memory_key_padding_mask).transpose(0, 1)
        if self.motion_pred:
            tgt = torch.cat([tgt, tgt], dim=0)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


class Objectselectattention(nn.Module):
    def __init__(self, d_model=256, 
                 dropout=0.1,
                 n_heads=8,
                 score_thresd=0.2,
                ):
        super().__init__()
        # self attention
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        # ffn
        self.class_emb = nn.Linear(d_model, 2)
        self.score_thresd = score_thresd
    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    @autocast(enabled=False)
    def forward(self,
                # for tgt
                tgt: Optional[Tensor],  # nq, bs, d_model
                tgt_query_pos: Optional[Tensor] = None,  # pos for query. MLP(Sine(pos))
                # sa
                self_attn_mask: Optional[Tensor] = None,  # mask used for self-attention
                num_object_query: int = 0,
                training: bool = False,
                num_new_object: int = 0,
                ):
        """
        Input:
            - tgt/tgt_query_pos: nq, bs, d_model
            -
        """
        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)
        # filter
        tgt = tgt[:num_object_query]
        cls_scores_return = self.class_emb(tgt).transpose(0,1) 
        cls_scores = cls_scores_return.sigmoid()
        num_filter_query = (cls_scores[:, :, 0] > self.score_thresd).sum(dim=1)
        num_object_query_list = [int(num_object_query*0.2), int(num_object_query*0.4), int(num_object_query*0.6), int(num_object_query*0.8), num_object_query]
        if self.training:
            num_filter_query = num_new_object 
        for i in num_object_query_list:
            if i >= num_filter_query:
                top_k = i
                break
        topk_proposals = torch.topk(cls_scores[:, :, 0], top_k, dim=1)[1] #只需要cell概率最大的topk个值
        cls_scores_return = torch.gather(cls_scores_return, 1,
                                   topk_proposals.unsqueeze(-1).repeat(1, 1, 2)) 
        # tgt_output = torch.gather(tgt, 1,
        #                           topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))  # unsigmoid
        if training:
            return topk_proposals, cls_scores_return, cls_scores
        else:
            return topk_proposals


