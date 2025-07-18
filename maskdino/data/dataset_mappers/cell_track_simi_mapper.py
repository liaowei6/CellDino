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
from torchvision import transforms
from PIL import Image
from pycocotools import mask as coco_mask



def annotations_to_instances(annos, image_size, mask_format = "polygon"):
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

    object_id = [int(obj["object_id"]) for obj in annos]
    object_id = torch.tensor(object_id, dtype=torch.int64)
    target.gt_ids = object_id

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

    if len(annos) and "crop_pos" in annos[0]:
        mask_pos = np.stack(
            [obj["crop_pos"] for obj in annos]
        )
        target.gt_mask_pos = mask_pos
        mask_size = np.stack(
            [obj["mask_size"] for obj in annos]
        )
        target.gt_mask_size = mask_size
    if len(annos) and "keypoints" in annos[0]:
        kpts = [obj.get("keypoints", []) for obj in annos]
        target.gt_keypoints = Keypoints(kpts)

    return target

def filter_empty_instances(
    instances, by_box=True, by_mask=True, box_threshold=1e-5, return_mask=False, moists=None, first=True,
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
        r.append(instances.gt_masks.nonempty())
    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        x = x.to(torch.bool)
        m = m & x
    if return_mask:
        return instances[m], m
    exit_ids = instances[m].gt_ids
    moists_new = moists.copy()
    if moists != None:
        for i in moists:
            if first:
                if int(i) not in exit_ids:
                    moists_new.pop(i)
            else:
                subset_tensor = torch.tensor(moists[i], dtype=torch.int64, device=exit_ids.device)
                if torch.all(torch.eq(subset_tensor.unsqueeze(1), exit_ids).any(dim=1)):
                    moists_new[int(i)] = moists[i]
                moists_new.pop(i)
    return instances[m], moists_new

def filter_empty_instances_simi(
    instances,  by_mask=True, box_threshold=1e-5, return_mask=False, moists=None, first=True,
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
    r = []
    h, w = instances.image_size
    if instances.has("gt_masks"):
        r.append(instances.gt_masks.nonempty())
        

    # TODO: can also filter visible keypoints

    if not r:
        return instances
    m = r[0]
    for x in r[1:]:
        x = x.to(torch.bool)
        m = m & x
    if return_mask:
        return instances[m], m
    exit_ids = instances[m].gt_ids
    return instances[m], moists

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
        T.FixedSizeCrop(crop_size=(image_size, image_size), pad_value=0.0 , seg_pad_value = 0.0),
        
    ])

    return augmentation

def normalize(img, lower=0.01, upper=99.99, low_value=0, up_value=255):
    lower_perc = np.percentile(img, lower)
    upper_perc = np.percentile(img, upper)
    return np.interp(img, (lower_perc, upper_perc), (low_value, up_value))

class CellTrackDatasetMapper_simi:
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
        is_crop=False,
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
        self.is_crop = is_crop
        self.transformer_blur = albumentations.Blur(p=0.3)
        self.color_jitter = transforms.ColorJitter(
            brightness=0.2,  # 亮度调整范围
            contrast=0.2,    # 对比度调整范围
            saturation=0.2,  # 饱和度调整范围
            hue=0.1          # 色调调整范围
        )
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
        #假设读取的范围为0-256
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        image_strong = image.copy() 
        image_second = tifffile.imread(dataset_dict["file_name_second"])
        image_second = np.repeat(image_second[:, :, np.newaxis], 3, axis=2)
        image_second_strong = image_second.copy()
        # image = np.broadcast_to(image[:, :, np.newaxis], (256, 256, 3))
        # image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        # 对图像进行归一化
        # 设计随机灰度映射
        random_low = np.random.random() * 50
        random_up = np.random.random() * 50 + 205
        image = normalize(image.astype("float32"), lower=1.0, upper=99.0, low_value=random_low, up_value=random_up).astype("float32")
        image_second = normalize(image_second.astype("float32"), lower=1.0, upper=99.0, low_value=random_low, up_value=random_up).astype("float32")
        image_strong = normalize(image_strong.astype("float32"), lower=1.0, upper=99.0, low_value=random_low, up_value=random_up).astype("float32")
        image_second_strong = normalize(image_second_strong.astype("float32"), lower=1.0, upper=99.0, low_value=random_low, up_value=random_up).astype("float32")
        #image = normalize(image.astype("float32"), lower=1.0, upper=99.0).astype("float32")
        #随机加入高斯噪声
        noise = np.random.normal(0, 25, image.shape[0:2]).astype("float32")
        noise = np.expand_dims(noise, axis=-1)
        noise = np.repeat(noise, 3, -1)
        noise_strong = np.random.normal(0, 50, image.shape[0:2]).astype("float32")
        noise_strong = np.expand_dims(noise_strong, axis=-1)
        noise_strong = np.repeat(noise_strong, 3, -1)
        image_strong = image_strong + noise_strong
        image_second_strong = image_second_strong + noise_strong
        image_strong = np.clip(image_strong, 0, 255)
        image_second_strong = np.clip(image_second_strong, 0, 255)
        p_1 = np.random.random()
        if p_1 > 0.5:
            image = image + noise
            image_second = image_second + noise
        image = np.clip(image, 0, 255)
        image_second = np.clip(image_second, 0, 255)

        parent_dir = "/".join(dataset_dict["file_name"].split("/")[:-2])
        utils.check_image_size(dataset_dict, image)
        
        # TODO: get padding mask
        # by feeding a "segmentation mask" to the same transforms
        padding_mask = np.ones(image.shape[:2])
        image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        image_second = transforms.apply_image(image_second)
        image_strong = transforms.apply_image(image_strong)
        image_second_strong = transforms.apply_image(image_second_strong)

        #加入随机图像模糊以及颜色扰乱
        image_strong = self.transformer_blur(image = image_strong)["image"].astype("uint8")
        image_second_strong = self.transformer_blur(image = image_second_strong)["image"].astype("uint8")
        image_strong = self.color_jitter(Image.fromarray(image_strong.astype("uint8")))
        image_second_strong = self.color_jitter(Image.fromarray(image_second_strong.astype("uint8")))
        image_strong = np.clip(np.array(image_strong), 0, 255).astype("float32")
        image_second_strong = np.clip(np.array(image_second_strong), 0, 255).astype("float32")
        # the crop transformation has default padding value 0 for segmentation, but fixedcropsize has default padding value 255 for segmentation
        padding_mask = transforms.apply_segmentation(padding_mask)
        padding_mask = ~ padding_mask.astype(bool)
        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict_second = {}
        dataset_dict_second["image"] = torch.as_tensor(np.ascontiguousarray(image_second.transpose(2, 0, 1)))
        dataset_dict_second["image_strong"] = torch.as_tensor(np.ascontiguousarray(image_second_strong.transpose(2, 0, 1)))
        dataset_dict_second["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict_second["height"] = dataset_dict["height"]
        dataset_dict_second["width"] = dataset_dict["width"]
        # dataset_dict_second["norm_image"] =torch.as_tensor(np.ascontiguousarray(norm_image_second))

        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = torch.as_tensor(np.ascontiguousarray(padding_mask))
        dataset_dict["image_strong"] = torch.as_tensor(np.ascontiguousarray(image_strong.transpose(2, 0, 1)))
        # dataset_dict["norm_image"] = torch.as_tensor(np.ascontiguousarray(norm_image))
        dataset_dict.pop("file_name")
        dataset_dict.pop("file_name_second")
        moists = dataset_dict["moists"]
        dataset_dict.pop("moists")

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
            annos = []
            annos_second = []
            # annotations_orgin = dataset_dict["annotations"]
            # dataset_dict.pop("annotations")
            for obj in dataset_dict.pop("annotations"):
                if obj["first"]:
                    annos.append(transform_instance_annotations(obj, transforms, image_shape, parent_dir,crop=self.is_crop))
                else:
                    annos_second.append(transform_instance_annotations(obj, transforms, image_shape, parent_dir,crop=self.is_crop, first=False))
            # NOTE: does not support BitMask due to augmentation
            # Current BitMask cannot handle empty objects
            instances = annotations_to_instances(annos, image_shape)
            instances_second = annotations_to_instances(annos_second, image_shape)
            # After transforms such as cropping are applied, the bounding box may no longer
            # tightly bound the object. As an example, imagine a triangle object
            # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
            # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
            # the intersection of original bounding box and the cropping box.
            if not instances.has('gt_masks'):  # this is to avoid empty annotation
                instances.gt_masks = PolygonMasks([])
            #instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            # Need to filter empty instances first (due to augmentation)
            instances, moists = filter_empty_instances(instances, moists=moists)
            instances_second, moists = filter_empty_instances(instances_second, moists=moists, first=False)
            # Generate masks from polygon
            h, w = instances.image_size
            if hasattr(instances, 'gt_masks'):
                gt_masks = instances.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances.gt_masks = gt_masks
            if hasattr(instances_second, 'gt_masks'):
                gt_masks = instances_second.gt_masks
                gt_masks = convert_coco_poly_to_mask(gt_masks.polygons, h, w)
                instances_second.gt_masks = gt_masks

            dataset_dict["instances"] = instances
            dataset_dict["moists"] = moists
            dataset_dict_second["instances"] = instances_second
            dataset_dict_second["moists"] = moists
            dataset_dict["second"] = dataset_dict_second
        return dataset_dict

def transform_instance_annotations(
    annotation, transforms, image_size, parent_path, keypoint_hflip_indices=None, crop = False, first=True
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
    crop_size = 256
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
        #segm = torch.load(segm_path)
        if isinstance(segm, list):
            # polygons
            # polygons = [np.asarray(p).reshape(-1, 2) for p in segm]
            if first:
                polygons = transforms.apply_polygons([np.asarray(p).reshape(-1, 2) for p in segm])
                annotation["segmentation"] = [
                    p.reshape(-1) for p in polygons
                ]
            else:
                r = cv2.boundingRect(np.asarray(segm[0]).reshape(-1, 2).astype(np.float32))
                p = np.array([[r[0],r[1]], [r[0]+r[2],r[1]], [r[0],r[1]+r[3]], [r[0]+r[2],r[1]+r[3]]])
                polygons = transforms.apply_polygons([p])
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
        elif isinstance(segm, np.ndarray):  #从数据增强后的mask中获取边框
            mask = transforms.apply_segmentation(segm)
            assert tuple(mask.shape[:2]) == image_size
            mask[mask > 0] = 1
            annotation["segmentation"] = mask
            kernel = np.ones((3, 3), np.uint8)
            kernel[0,0] = 0
            kernel[0,2] = 0
            kernel[2,0] = 0
            kernel[2,2] = 0
            mask_open =  cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(mask_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                contours = contours[0].astype(np.float32) #(points,1,2)
                r = cv2.minAreaRect(contours)
                obbox = np.array([r[0][0], r[0][1], r[1][1], r[1][0], 90 - r[2]])     
            else:
                obbox = np.zeros(5)
            annotation["bbox"] = obbox
            annotation["bbox_mode"] = BoxMode.XYWHA_ABS
            if crop:
                center_pos = [int(obbox[0]), int(obbox[1])]
                crop_pos = [0, crop_size, 0, crop_size]
                if image_size[1] - center_pos[0] < crop_size:
                    crop_pos[0] = image_size[1] - crop_size
                    crop_pos[1] = image_size[1] 
                elif center_pos[0] > crop_size:
                    crop_pos[0] = center_pos[0] - int(crop_size/2)
                    crop_pos[1] = center_pos[0] + int(crop_size/2)
                if image_size[0] - center_pos[1] < crop_size:
                    crop_pos[2] = image_size[0] - crop_size
                    crop_pos[3] = image_size[0]
                elif center_pos[1] > crop_size:
                    crop_pos[2] = center_pos[1] - int(crop_size/2)
                    crop_pos[3] = center_pos[1] + int(crop_size/2)
                annotation["segmentation"] = mask[crop_pos[2]:crop_pos[3], crop_pos[0]:crop_pos[1]]
                annotation["crop_pos"] = crop_pos
                annotation["mask_size"] = image_size
        else:
            raise ValueError(
                "Cannot transform segmentation of type '{}'!"
                "Supported types are: polygons as list[list[float] or ndarray],"
                " COCO-style RLE as a dict.".format(type(segm))
            )


    return annotation

