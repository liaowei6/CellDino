from detectron2.data.dataset_mapper import DatasetMapper
import copy
import numpy as np
import torch
import cv2
import tifffile
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T

def normalize(img, lower=0.01, upper=99.99):
    lower_perc = np.percentile(img, lower)
    upper_perc = np.percentile(img, upper)
    return np.interp(img, (lower_perc, upper_perc), (0, 255))


def build_transform_gen(image):
    """
    Create a list of default :class:`Augmentation` from config.
    Now it includes resizing and flipping.
    Returns:
        list[Augmentation]
    """
    h,w = image.shape[:2]
    image_size = max(h,w)
    augmentation = []
    if h>w:
        pad_size = (image_size - w) % 32 
        augmentation.append(T.PadTransform(pad_size//2+pad_size%2, 0, pad_size//2, 0))
        pad_map = torch.tensor([pad_size//2+pad_size%2, 0, pad_size//2, 0])
    else:
        pad_size = (image_size - h) % 32 
        augmentation.append(T.PadTransform(0, pad_size//2+pad_size%2, 0, pad_size//2))
        pad_map = torch.tensor([0, pad_size//2+pad_size%2, 0, pad_size//2])
    return augmentation, pad_map

class DatasetMapper_test(DatasetMapper):
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = tifffile.imread(dataset_dict["file_name"])
        image = np.repeat(image[:, :, np.newaxis], 3, axis=2)
        #输出真实框进行可视化
        # boxes = (np.stack(obj["bbox"] for obj in dataset_dict["annotations"]))
        # dataset_dict["bbox"] = torch.tensor(boxes)
        #对图像进行归一化
        image = normalize(image.astype("float32"), lower=1.0, upper=99.0).astype("float32")
        image = cv2.bilateralFilter(image, 9, 75, 75)  #降噪
        dataset_dict["image_raw"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        utils.check_image_size(dataset_dict, image)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None
        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms = self.augmentations(aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        image_shape = image.shape[:2]  # h, w\
        #将图像等比例放缩后长宽都填充为可以被32整除的大小
        augmentation_pad, pad_map= build_transform_gen(image)
        image, transforms_pad = T.apply_transform_gens(augmentation_pad, image)
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))
        dataset_dict["padding_mask"] = pad_map
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))
        if "annotations" in dataset_dict:
            dataset_dict["num"] = len(dataset_dict["annotations"])
        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.

            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict