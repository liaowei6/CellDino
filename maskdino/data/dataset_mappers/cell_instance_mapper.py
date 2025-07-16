# ------------------------------------------------------------------------
# Copyright (c) 2022 IDEA. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Mask2Former https://github.com/facebookresearch/Mask2Former by Feng Li.
import copy
import logging
import os
import numpy as np
import torch
import tifffile
import cv2
import pycocotools.mask as mask_util
from detectron2.config import configurable
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.data.transforms import TransformGen
from detectron2.structures import (
    BitMasks,
    Boxes,
    BoxMode,
    Instances,
    Keypoints,
    PolygonMasks,
    RotatedBoxes,
    polygons_to_bitmask,
)


from pycocotools import mask as coco_mask

__all__ = ["CellinstanceDatasetMapper"]


def annotations_to_instances(annos, image_size):
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
            [BoxMode.convert(obj["bbox"], obj["bbox_mode"], BoxMode.XYXY_ABS) for obj in annos]
        )
        if len(annos)
        else np.zeros((0, 4))
    )
    target = Instances(image_size)
    target.gt_boxes = Boxes(boxes)

    classes = [int(obj["category_id"]) for obj in annos]
    classes = torch.tensor(classes, dtype=torch.int64)
    target.gt_classes = classes

    if len(annos) and "segmentation" in annos[0]:
        segms = [obj["segmentation"] for obj in annos]
        masks = []
        for segm in segms:
            masks.append(segm)
        # torch.from_numpy does not support array with negative stride.
        masks = torch.stack([torch.from_numpy(np.ascontiguousarray(x)) for x in masks])
        target.gt_masks = masks

    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target

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
            min_scale=min_scale, max_scale=max_scale, target_height=image_size, target_width=image_size
        ),
        T.FixedSizeCrop(crop_size=(image_size, image_size),seg_pad_value=0),
    ])

    return augmentation

def normalize(img, lower=0.01, upper=99.99, low_value=0, up_value=255):
    lower_perc = np.percentile(img, lower)
    upper_perc = np.percentile(img, upper)
    return np.interp(img, (lower_perc, upper_perc), (low_value, up_value))

class CellinstanceDatasetMapper:
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
        image = tifffile.imread(dataset_dict["file_name"])
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        # image = np.broadcast_to(image[:, :, np.newaxis], (256, 256, 3))
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # 对图像进行归一化
        # 设计随机灰度映射
        random_low = np.random.random() * 50
        random_up = np.random.random() * 50 + 205
        #随机加入高斯噪声
        noise = np.random.normal(0, 20, image.shape[0:2]).astype("float32")
        noise = np.expand_dims(noise, axis=-1)
        noise = np.repeat(noise, 3, -1)
        p_1 = np.random.random()
        if p_1 > 0.5:
            image = image + noise

        image = normalize(image.astype("float32"), lower=1.0, upper=99.0, low_value=random_low, up_value=random_up).astype("float32")
        parent_dir = "/".join(dataset_dict["file_name"].split("/")[:-2])
        utils.check_image_size(dataset_dict, image)

        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])

        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        # the crop transformation has default padding value 0 for segmentation
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
            instances = filter_empty_instances(instances)
            # Generate masks from polygon
            # h, w = instances.image_size
            # if hasattr(instances, 'gt_masks'):
            #     gt_masks = instances.gt_masks
            #     gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
            #     instances.gt_masks = gt_masks

            dataset_dict["instances"] = instances

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
    bbox = BoxMode.convert(annotation["bbox"], annotation["bbox_mode"], BoxMode.XYXY_ABS)
    # clip transformed bbox to image size
    bbox = transforms.apply_box(np.array([bbox]))[0].clip(min=0)
    annotation["bbox"] = np.minimum(bbox, list(image_size + image_size)[::-1])
    annotation["bbox_mode"] = BoxMode.XYXY_ABS

    if "instance_segmentation" in annotation:
        # each instance contains 1 or more polygons
        segm_path = annotation["instance_segmentation"]
        segm = torch.load(segm_path)
        if isinstance(segm, list):
            # polygons
            polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            annotation["segmentation"] = [
                p.reshape(-1) for p in transforms.apply_polygons(polygons)
            ]
        elif isinstance(segm, dict):
            # RLE
            mask = mask_util.decode(segm)
            mask = transforms.apply_segmentation(mask)
            assert tuple(mask.shape[:2]) == image_size
            annotation["segmentation"] = mask 
        elif isinstance(segm, np.ndarray):
            mask = transforms.apply_segmentation(segm)
            assert tuple(mask.shape[:2]) == image_size
            mask[mask > 0] = 1
            kernel = np.ones((3, 3), np.uint8)
            kernel[0,0] = 0
            kernel[0,2] = 0
            kernel[2,0] = 0
            kernel[2,2] = 0
            mask_open =  cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            if not np.any(mask_open):
                annotation["bbox"] = np.array([0,0,0,0], dtype=float)
            else:
                rows, cols = np.where(mask_open)
                y_min, y_max = np.min(rows), np.max(rows)
                x_min, x_max = np.min(cols), np.max(cols)
                annotation["bbox"] = np.array([x_min, y_min, x_max, y_max], dtype=float)
            annotation["segmentation"] = mask
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )


    return annotation