# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li.
import copy
import logging
import os
import numpy as np
import cv2
import torch
import tifffile
import albumentations
from skimage.exposure import equalize_adapthist
import pycocotools.mask as mask_util
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.utils.visualizer import Visualizer
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import (
    BitMasks,
    Boxes,
    RotatedBoxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)


from pycocotools import mask as coco_mask

__all__ = ["CellinstanceObboxDatasetMapper"]

#固定随机数种子用以测试
seed_value = 30
np.random.seed(30)

def annotations_to_instances(annos, image_size, mask_format="polygon"):
    """
    Create an :class:`Instances` object used by the models,
    from instance annotations in the dataset dict.

    Args:
        annos (list[dict]): a list of instance annotations in one image, each
            element for one instance.
        image_size (tuple): height, width

    Returns:
        Instances:
            It will contain fields "gt_boxes", "gt_classes",
            "gt_masks", "gt_keypoints", if they can be obtained from `annos`.
            This is the format that builtin models expect.
    """
    boxes = (
        np.stack(
            [obj["bbox"] for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 5))
    )
    target = Instances(image_size) 
    target.gt_boxes = RotatedBoxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        if mask_format == "polygon":
            try:
                masks = PolygonMasks(segms)
            except ValueError as e:
                raise ValueError(
                    "Failed to use mask_format=='polygon' from the given annotations!"
                ) from e
        else:
            assert mask_format == "bitmask", mask_format
            masks = []
            for segm in segms:
                if isinstance(segm, list):
                    # polygon
                    masks.append(polygons_to_bitmask(segm, *image_size))
                elif isinstance(segm, dict):
                    # COCO RLE
                    masks.append(mask_util.decode(segm))
                elif isinstance(segm, np.ndarray):
                    assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                        segm.ndim
                    )
                    # mask array
                    masks.append(segm)
                else:
                    raise ValueError(
                        "Cannot convert segmentation of type '{}' to BitMasks!"
                        "Supported types are: polygons as list[list[float] or ndarray],"
                        " COCO-style RLE as a dict, or a binary segmentation mask "
                        " in a 2D numpy array of shape HxW.".format(type(segm))
                    )
            # torch.from_numpy does not support array with negative stride.
            masks = BitMasks(
                torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
            )
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target

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

def filter_empty_instances(
    instances, by_box=True, by_mask=True, box_threshold=1e-5, return_mask=False
):
    """
    Filter out empty instances in an `Instances` object.

    Args:
        instances (Instances):
        by_box (bool): whether to filter out instances with empty boxes
        by_mask (bool): whether to filter out instances with empty masks
        box_threshold (float): minimum width and height to be considered non-empty
        return_mask (bool): whether to return boolean mask of filtered instances

    Returns:
        Instances: the filtered instances.
        tensor[bool], optional: boolean mask of filtered instances
    """
    assert by_box or by_mask
    r = []
    if by_box:
        r.append(instances.gt_boxes.nonempty(threshold=box_threshold))
    if instances.has("gt_masks") and by_mask:
        r.append(instances.gt_masks.flatten(1).any(dim=1))

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        x = x.to(torch.bool)
        m = m & x
    if return_mask:
        return instances[m], m
    return instances[m]

# def filter_empty_instances(instance):
#     n, h, w = instance.shape
#     is_all_zero = torch.all(instance.view(n, -1) == 0, dim=1)
#     filtered_instance = instance[~is_all_zero]
#     return filtered_instance

def build_transform_gen(cfg, is_train):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    assert is_train, "Only support training augmentation"
    image_size = cfg.INPUT.IMAGE_SIZE
    min_scale = cfg.INPUT.MIN_SCALE
    max_scale = cfg.INPUT.MAX_SCALE

    augmentation = []

    if cfg.INPUT.RANDOM_FLIP != "none":
        augmentation.append(
            T.RandomFlip(
                horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal",
                vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
            )
        )

    augmentation.extend([
        T.RandomRotation(angle=[0.0, 360.0]),
        T.ResizeScale(
            min_scale=1.0, max_scale=3.0, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size), seg_pad_value = 0.0),
        
    ])

    return augmentation

def normalize(img, lower=0.01, upper=99.99):
    lower_perc = np.percentile(img, lower)
    upper_perc = np.percentile(img, upper)
    return np.interp(img, (lower_perc, upper_perc), (0, 255))

class LiveCellinstanceObboxDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer.

    This dataset mapper applies the same transformation as DETR for COCO panoptic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        tfm_gens,
        image_format,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            tfm_gens: data augmentation
            image_format: an image format supported by :func:`detection_utils.read_image`.
        """
        self.tfm_gens = tfm_gens
        logging.getLogger(__name__).info(
            "[CellInstanceDatasetMapper] Full TransformGens used in training: {}".format(str(self.tfm_gens))
        )

        self.img_format = image_format
        self.is_train = is_train
        self.transformer_blur = albumentations.Blur(p=0.3)
    @classmethod
    def from_config(cls, cfg, is_train=True):
        # Build augmentation
        tfm_gens = build_transform_gen(cfg, is_train)

        ret = {
            "is_train": is_train,
            "tfm_gens": tfm_gens,
            "image_format": cfg.INPUT.FORMAT,
        }
        return ret

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image("datasets/live_cell/images/livecell_train_val_images/" + dataset_dict["file_name"], format=self.img_format)
        #对图像进行归一化
        image = normalize(image.astype("float32"), lower=1.0, upper=99.0).astype("float32")
        parent_dir = "/".join(dataset_dict["file_name"].split("/")[:-2])
        utils.check_image_size(dataset_dict, image)
        
        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        #随机加入高斯噪声
        noise = np.random.normal(0, 5, image.shape[0:2]).astype("float32")
        noise = np.expand_dims(noise, axis=-1)
        noise = np.repeat(noise, 3, -1)
        p_1 = np.random.random()
        if p_1 > 0.5:
            image = image + noise
        #加入随机图像模糊
        # image = self.transformer_blur(image = image)["image"]
        image = np.clip(image, 0, 255)
        # the crop transformation has default padding value 0 for segmentation, but fixedcropsize has default padding value 255 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                # Let's always keep mask
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            current_dir = os.path.abspath(os.path.dirname(__file__))
            annos = [
                transform_instance_annotations(obj, transforms, image_shape, parent_dir)
                for obj in dataset_dict.pop("annotations")
            ]
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = annotations_to_instances(annos, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'):  # this is to avoid empty annotation
                instances.gt_masks = PolygonMasks([])
            #instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances = utils.filter_empty_instances(instances)
            dataset_dict["instances"] = instances
            # Generate masks from polygon
            # h, w = instances.image_size
            # if hasattr(instances, 'gt_masks'):
            #     gt_masks = instances.gt_masks
            #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            #     instances.gt_masks = gt_masks
        # 可视化训练样本
        # scores = torch.ones_like(instances.gt_classes)
        # instances.scores = scores
        # instances.pred_classes = instances.gt_classes
        # instances.pred_masks = instances.gt_masks
        # instances.pred_boxes = instances.gt_boxes
        # visualizer = Visualizer(image)
        # item_vis_output = visualizer.draw_instance_predictions(instances.to(torch.device("cpu")))
        # item_vis_output.save("output/images_train_vis/" + dataset_dict["file_name"])
        return dataset_dict


def transform_instance_annotations(
    annotation, transforms, image_size, parent_path, keypoint_hflip_indices=None
):
    """
    Apply transforms to box, segmentation and keypoints annotations of a single instance.

    It will use `transforms.apply_box` for the box, and
    `transforms.apply_coords` for segmentation polygons & keypoints.
    If you need anything more specially designed for each data structure,
    you'll need to implement your own version of this function or the transforms.

    Args:
        annotation (dict): dict of instance annotations for a single instance.
            It will be modified in-place.
        transforms (TransformList or list[Transform]):
        image_size (tuple): the height, width of the transformed image
        keypoint_hflip_indices (ndarray[int]): see `create_keypoint_hflip_indices`.

    Returns:
        dict:
            the same input dict with fields "bbox", "segmentation", "keypoints"
            transformed according to `transforms`.
            The "bbox_mode" field will be set to XYXY_ABS.
    """
    if isinstance(transforms, (tuple, list)):
        transforms = T.TransformList(transforms)
    # bbox is 1d (per-instance bounding box)
    # bbox = BoxMode.convert(annotation["bbox"], BoxMode.XYWHA_ABS, BoxMode.XYWHA_ABS)
    # clip transformed bbox to image size
    # bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    # annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    # annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm = annotation["segmentation"]
        if isinstance(segm, list):            
            polygons = transforms.apply_polygons([np.asarray(p).reshape(-1, 2) for p in segm])
            annotation["segmentation"] = [
                p.reshape(-1) for p in polygons
            ]
            if len(polygons) > 0:
                r = cv2.minAreaRect(polygons[0].astype(np.float32))
                obbox = np.array([r[0][0], r[0][1], r[1][1], r[1][0], 90 - r[2]])     
            else:
                obbox = np.zeros(5)
            annotation["bbox"] = obbox
            annotation["bbox_mode"] = BoxMode.XYWHA_ABS
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask 

        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )


    return annotation