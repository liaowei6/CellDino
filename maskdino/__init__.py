# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li and Hao Zhang.
# ------------------------------------------------------------------------------
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskdino_config

# dataset loading
from .data.dataset_mappers.coco_instance_new_baseline_dataset_mapper import COCOInstanceNewBaselineDatasetMapper
from .data.dataset_mappers.coco_panoptic_new_baseline_dataset_mapper import COCOPanopticNewBaselineDatasetMapper
from .data.dataset_mappers.detr_dataset_mapper import DetrDatasetMapper
from .data.dataset_mappers.cell_instance_mapper import CellinstanceDatasetMapper
from .data.dataset_mappers.cell_instance_obbox_mapper import CellinstanceObboxDatasetMapper
from .data.dataset_mappers.cell_track_mapper import CellTrackDatasetMapper 
from .data.dataset_mappers.cell_track_simi_mapper import CellTrackDatasetMapper_simi
from .data.dataset_mappers.live_cell_instance_obbox_mapper import LiveCellinstanceObboxDatasetMapper
from .data.dataset_mappers.dataset_mapper_test import DatasetMapper_test
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)
# models
from .maskdino import MaskDINO
from .ctdino import CtDINO
from .cgtdino import CgtDINO
# from .data.datasets_detr import coco
from .test_time_augmentation import SemanticSegmentorWithTTA

# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator
from .evaluation.obbox_evaluator import ObboxEvaluator
from .evaluation.cell_evaluation import CellbboxEvaluator
from .evaluation.inference import inference_on_dataset,inference_on_dataset_submit,inference_on_dataset_cell,inference_on_dataset_track,inference_on_dataset_track_submit
# util
from .utils import box_ops, misc, utils

#Tracker
from .tracker import Tracker, Track

#Trainer
from .trainer.trainer import DefaultTrainer_cell, AMPTrainer_cell, AMPTrackTrainer_cell

#Register
from .trainer.coco import register_coco_instances
