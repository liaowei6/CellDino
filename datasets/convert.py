import torch
import os
import re
import pandas as pd
import numpy as np
from tifffile import imread, imsave
import cv2

def get_indices_pandas(data, background_id=0):
    """
    Extracts for each mask id its positions within the array.
    Args:
        data: a np. array with masks, where all pixels belonging to the
            same masks have the same integer value
        background_id: integer value of the background

    Returns: data frame: indices are the mask id , values the positions of the mask pixels

    """
    if data.size < 1e9:  # aim for speed at cost of high memory consumption
        masked_data = data != background_id
        flat_data = data[masked_data]  # d: data , mask attribute
        dummy_index = np.where(masked_data.ravel())[0]
        df = pd.DataFrame.from_dict({"mask_id": flat_data, "flat_index": dummy_index})
        df = df.groupby("mask_id").apply(
            lambda x: np.unravel_index(x.flat_index, data.shape)
        )
    else:  # aim for lower memory consumption at cost of speed
        flat_data = data[(data != background_id)]  # d: data , mask attribute
        dummy_index = np.where((data != background_id).ravel())[0]
        mask_indices = np.unique(flat_data)
        df = {"mask_id": [], "index": []}
        data_shape = data.shape
        for mask_id in mask_indices:
            df["index"].append(
                np.unravel_index(dummy_index[flat_data == mask_id], data_shape)
            )
            df["mask_id"].append(mask_id)
        df = pd.DataFrame.from_dict(df)
        df = df.set_index("mask_id")
        df = df["index"].apply(lambda x: x)  # convert to same format as for other case
    return df

def generat_gt_lineage_and_bounding_boxes(gt_seg_path, gt_track_path, box_path, instance_mask_path, res_path=None):
    """
    Use the GT segmentation and the GT tracking annotations to create an annotated dataset.
    Args:
        gt_seg_path (string): path to the GT annotation
        gt_track_path (string): path to the GT tracking annotation
        res_path (string): path where to store the fused annotations

    Returns:

    """
    if res_path is None:
        res_path = gt_seg_path

    if not os.path.exists(box_path):
            os.makedirs(box_path)

    if not os.path.exists(instance_mask_path):
            os.makedirs(instance_mask_path)

    # gt_seg_files = {
    #     int(re.findall("\d+", file)[0]): os.path.join(gt_seg_path, file)
    #     for file in os.listdir(gt_seg_path)
    #     if file.endswith(".tif")
    # }
    gt_track_files = {
        int(re.findall("\d+", file)[0]): os.path.join(gt_track_path, file)
        for file in os.listdir(gt_track_path)
        if file.endswith(".tif")
    }
    for time in sorted(gt_track_files.keys()):
        #gt_seg_mask = imread(gt_seg_files[time])
        gt_track_mask = imread(gt_track_files[time])
        fused_mask = np.zeros_like(gt_track_mask)
        seg_mask_indices = get_indices_pandas(gt_track_mask)
        for mask_id, mask_indices in seg_mask_indices.items():
            instance_mask_file = os.path.join(instance_mask_path, str(time).zfill(4) + "_" + str(mask_id) + ".pt")
            box_file = os.path.join(box_path, str(time).zfill(4) + "_" + str(mask_id) + ".pt")
            instance_mask = np.zeros_like(gt_track_mask, dtype= "uint8")
            gt_obj_ids = np.unique(gt_track_mask[mask_indices])
            gt_obj_ids = gt_obj_ids[gt_obj_ids > 0]
            if len(gt_obj_ids) == 1:  # use st mask as mapped to single obj
                # top_left_y = np.amin(mask_indices[0])
                # top_left_x = np.amin(mask_indices[1])
                # down_right_y = np.amax(mask_indices[0])
                # down_right_x = np.amax(mask_indices[1])
                gt_id = gt_obj_ids[0]
                #fused_mask[mask_indices] = gt_id
                fused_mask[mask_indices] = 255
                instance_mask[mask_indices] = 1
                contours, _ = cv2.findContours(instance_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
                contours = contours[0].astype(np.float32) #(points,1,2)
                #获取最小外接矩形
                #r = cv2.minAreaRect(contours[0])
                r = cv2.minAreaRect(contours)
                box = np.array([gt_id, r[0][0], r[0][1], r[1][1], r[1][0], 90 - r[2]])  #(x, y, w, h, theta)
                # box = np.array([gt_id, top_left_x, top_left_y, down_right_x, down_right_y])
                torch.save(instance_mask, instance_mask_file)
                torch.save(box, box_file)
                # no gt mask == false positive -> don't add
        # file_name = os.path.split(gt_track_files[time])[1]
        # if not os.path.exists(res_path):
        #     os.makedirs(res_path)
        #imsave(os.path.join(res_path, file_name), fused_mask, compression=("ZSTD", 1))
        # imsave(os.path.join(res_path, file_name), fused_mask.astype("uint8"),)


gt_seg_path = "datasets/Fluo-C2DL-Huh7/02/02_GT/SEG"
gt_track_path = "datasets/Fluo-C2DL-Huh7/02/02_GT/TRA"
box_path = "datasets/Fluo-C2DL-Huh7/02/02_GT/bounding_boxes"
instance_mask_path = "datasets/Fluo-C2DL-Huh7/02/02_GT/instance_mask"
res_path = "datasets/Fluo-C2DL-Huh7/02/02_GT/masks"
# for i in range(12):
#     gt_seg_path = None
#     gt_track_path = f"datasets/deepcell/test/batch_{i}/SEG"
#     box_path = f"datasets/deepcell/test/batch_{i}/bounding_boxes"
#     instance_mask_path = f"datasets/deepcell/test/batch_{i}/instance_mask"
#     res_path = None
#     generat_gt_lineage_and_bounding_boxes(gt_seg_path, gt_track_path, box_path, instance_mask_path, None)

