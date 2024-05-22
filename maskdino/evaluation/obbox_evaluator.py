# Copyright (c) Facebook, Inc. and its affiliates.
import contextlib
import copy
import io
import itertools
import json
import logging
import numpy as np
import os
import pickle
import platform
import re
import shutil
import subprocess
from pathlib import Path
from collections import OrderedDict
import pycocotools.mask as mask_util
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tabulate import tabulate

import detectron2.utils.comm as comm
from detectron2.config import CfgNode
from detectron2.data import MetadataCatalog
from detectron2.data.datasets.coco import convert_to_coco_json
from detectron2.evaluation.coco_evaluation import COCOEvaluator, _evaluate_predictions_on_coco
from detectron2.evaluation.fast_eval_api import COCOeval_opt
from detectron2.structures import Boxes, BoxMode, pairwise_iou, RotatedBoxes
from detectron2.utils.file_io import PathManager
from detectron2.utils.logger import create_small_table
from mmcv.ops import box_iou_rotated
from tifffile import imsave

def select_platform():
    """Selects for the cell tracking challenge measure the correct executables."""
    sys_type = platform.system()
    if sys_type == "Linux":
        return "Linux"
    if sys_type == "Windows":
        return "Win"
    if sys_type == "Darwin":
        return "Mac"
    raise ValueError("Platform not supported")

# modified from COCOEvaluator for instance segmetnat
class ObboxEvaluator(COCOEvaluator):
    """
    Evaluate AR for object proposals, AP for instance detection/segmentation, AP
    for keypoint detection outputs using COCO's metrics.
    See http://cocodataset.org/#detection-eval and
    http://cocodataset.org/#keypoints-eval to understand its metrics.
    The metrics range from 0 to 100 (instead of 0 to 1), where a -1 or NaN means
    the metric cannot be computed (e.g. due to no predictions made).

    In addition to COCO, this evaluator is able to support any bounding box detection,
    instance segmentation, or keypoint detection dataset.
    """

    def process(self, inputs, outputs):
        """
        Args:
            inputs: the inputs to a COCO model (e.g., GeneralizedRCNN).
                It is a list of dict. Each dict corresponds to an image and
                contains keys like "height", "width", "file_name", "image_id".
            outputs: the outputs of a COCO model. It is a list of dicts with key
                "instances" that contains :class:`Instances`.
        """
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["instances"] = instances_to_coco_json_cell(instances, input["image_id"], thresdhold=0.01)
                prediction["ctc"] = instances_to_ctc(instances, input["image_id"], input["file_name"], thresdhold = 0.01)
            if "proposals" in output:
                prediction["proposals"] = output["proposals"].to(self._cpu_device)
            if len(prediction) > 1:
                self._predictions.append(prediction)

        
    def process_test(self, inputs, outputs):
        for input, output in zip(inputs, outputs):
            prediction = {"image_id": input["image_id"]}

            if "instances" in output:
                instances = output["instances"].to(self._cpu_device)
                prediction["ctc"] = instances_to_ctc_test(instances, input["image_id"], input["file_name"], thresdhold = 0.01)
            if len(prediction) > 1:
                self._predictions.append(prediction)

    def _eval_obox_proposals(self, predictions):
        """
        Evaluate the obox proposals in predictions.
        Fill self._results with the metrics for "box_proposals" task.
        """
        proposals = []
        if self._output_dir:
            # Saving generated box proposals to file.
            # Predicted box_proposals are in XYXY_ABS mode.
            bbox_mode = BoxMode.XYWHA_ABS.value
            ids, boxes, objectness_logits = [], [], []
            for prediction in predictions:
                ids.append(prediction["image_id"])
                bbox = torch.tensor([instance['bbox'] for instance in prediction["instances"]])
                objectness_logit = torch.tensor([instance['score'] for instance in prediction["instances"]])
                boxes.append(bbox)
                objectness_logits.append(objectness_logit)
                proposal = {
                "boxes": bbox,
                "objectness_logits": objectness_logit,
                "image_id": prediction['image_id'],
            }
                proposals.append(proposal)
            proposal_data = {
                "boxes": boxes,
                "objectness_logits": objectness_logits,
                "ids": ids,
                "bbox_mode": bbox_mode,
            }
            with PathManager.open(os.path.join(self._output_dir, "box_proposals.pkl"), "wb") as f:
                pickle.dump(proposal_data, f)
            
        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating bbox proposals ...")
        res = {}
        areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        for limit in [100, 1000]:
            for area, suffix in areas.items():
                stats = _evaluate_box_proposals(proposals, self._coco_api, area=area, limit=limit)
                key = "AR{}@{:d}".format(suffix, limit)
                res[key] = float(stats["ar"].item() * 100)
        self._logger.info("Proposal metrics: \n" + create_small_table(res))
        self._results["bbox"] = res

    def _eval_predictions(self, predictions, img_ids=None):
        """
        Evaluate predictions. Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))
        tasks = self._tasks or self._tasks_from_predictions(coco_results)
        tasks = []
        if "ctc" in predictions[0]:
            tasks.append("ctc")
        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            dataset_id_to_contiguous_id = self._metadata.thing_dataset_id_to_contiguous_id
            all_contiguous_ids = list(dataset_id_to_contiguous_id.values())
            num_classes = len(all_contiguous_ids)
            assert min(all_contiguous_ids) == 0 and max(all_contiguous_ids) == num_classes - 1

            reverse_id_mapping = {v: k for k, v in dataset_id_to_contiguous_id.items()}
            for result in coco_results:
                category_id = result["category_id"]
                assert category_id < num_classes, (
                    f"A prediction has class={category_id}, "
                    f"but the dataset only has {num_classes} classes and "
                    f"predicted class id should be in [0, {num_classes - 1}]."
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info(
            "Evaluating predictions with {} COCO API...".format(
                "unofficial" if self._use_fast_impl else "official"
            )
        )
        for task in sorted(tasks):
            assert task in {"bbox", "segm", "keypoints", 'ctc'}, f"Got unknown task: {task}!"
            if task == "bbox":
                self._eval_obox_proposals(predictions)
                continue
            elif task == "ctc":
                self._eval_ctc(predictions)
                continue
            coco_eval = (
                _evaluate_predictions_on_coco(
                    self._coco_api,
                    coco_results,
                    task,
                    kpt_oks_sigmas=self._kpt_oks_sigmas,
                    use_fast_impl=self._use_fast_impl,
                    img_ids=img_ids,
                    max_dets_per_image=self._max_dets_per_image,
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            res = self._derive_coco_results(
                coco_eval, task, class_names=self._metadata.get("thing_classes")
            )
            self._results[task] = res

    def _eval_ctc(self, predictions):
        '''
        采用ctc官方代码进行测试
        '''
        masks = []
        index_dataset = []
        for prediction in predictions:
            masks.append(prediction["ctc"]["segmentation"])
            index_dataset.append(prediction["ctc"]["index_dataset"])
        result_path = "output/testing_dataset/"
        id = 1
        last_index = index_dataset[0]
        for index, mask in zip(index_dataset, masks):
            if index != last_index:
                last_index = index
                id = 1
            mask = mask.numpy()
            if len(predictions) == 16:
                mask_file = os.path.join(result_path + index + "_RES", "mask" + str(id + 75).zfill(3) + ".tif")
            else:
                mask_file = os.path.join(result_path + index + "_RES", "mask" + str(id + 82).zfill(3) + ".tif")
            id += 1
            imsave(mask_file, mask.astype(np.uint16),)
        scores_1 = calc_ctc_scores(Path(result_path + "01_RES"), Path(result_path + "01_GT"))
        scores_2 = calc_ctc_scores(Path(result_path + "02_RES"), Path(result_path + "02_GT"))
        scores = {}
        scores["DET"] = (scores_1["DET"] + scores_2["DET"]) / 2
        scores["SEG"] = (scores_1["SEG"] + scores_2["SEG"]) / 2
        self._results["ctc"] = scores

def calc_ctc_scores(result_dir, gt_dir):
    """
    Extracts all CTC measures (DET,SEG,TRA) using the CTC executables.
    Args:
        result_dir: posix path to the tracking directory
        gt_dir: posix path to the ground truth data directory

    Returns: directory containing the metric names and their scores

    """

    ctc_measure_dir = Path(__file__).parent / "ctc_metrics" /"CTC_eval"
    assert Path.exists(ctc_measure_dir), "missing folder with CTC measures"
    platform_name = select_platform()
    measure_files = [
        file
        for file in os.listdir(Path(ctc_measure_dir, platform_name))
        if "Measure" in file
    ]   #获取测试文件

    data_dir = gt_dir.parent
    data_set_id = gt_dir.name.split("_")[0]
    # copy results to same dir as gt path, as ctc measure executable expects results
    # and gt data to be in the same directory
    temp_dir = None
    default_res_dir = data_dir / (data_set_id + "_RES")  #获取结果所在文件夹
    if result_dir.as_posix() != default_res_dir.as_posix():

        if os.path.exists(default_res_dir):
            temp_dir = data_dir / ("temp")
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            shutil.move(default_res_dir.as_posix(), temp_dir.as_posix())

        shutil.move(result_dir.as_posix(), default_res_dir.as_posix())
    # ctc executable expects time points in file names to have a fixed length
    n_files = max(
        [
            int(re.findall(r"\d+", f)[0])
            for f in os.listdir(gt_dir / "TRA")
            if f.endswith("tif")
        ]
    )
    if n_files < 1000:
        n_digits = 3
    else:
        n_digits = 4

    for f in os.listdir(default_res_dir):
        if f.startswith("mask"):
            parts = f.split(".")
            mask_id = f.split(".")[0].split("mask")[1].zfill(n_digits)
            new_name = "mask" + mask_id + "." + parts[1]
            shutil.move(default_res_dir / f, default_res_dir / new_name)

    all_results = {}
    for measure_f in measure_files:
        if not os.path.exists(
            default_res_dir / "res_track.txt"
        ) and measure_f.startswith("TRA"):
            continue
        ctc_metric_path = Path(ctc_measure_dir, platform_name, measure_f)
        output = subprocess.Popen(
            [
                ctc_metric_path.as_posix(),
                data_dir.as_posix(),
                data_set_id,
                str(n_digits),
            ],
            stdout=subprocess.PIPE,
        )
        result, _ = output.communicate()
        print(result)
        metric_score = re.findall(r"\d\.\d*", result.decode("utf-8"))
        all_results[measure_f[:3]] = metric_score

    print(result_dir)
    # undo moving of result folder
    if result_dir.as_posix() != default_res_dir.as_posix():
        shutil.move(default_res_dir.as_posix(), result_dir.as_posix())
        if temp_dir is not None:
            shutil.move(temp_dir.as_posix(), default_res_dir.as_posix())

    for metric_name, score in all_results.items():
        assert len(score) == 1, f"error in extraction of {metric_name} measure"
    all_results = {
        metric_name: float(score[0]) for metric_name, score in all_results.items()
    }
    return all_results


def instances_to_coco_json_cell(instances, img_id, thresdhold = 0.01):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()

    has_mask = instances.has("pred_masks")
    if has_mask:
        # use RLE to encode the masks, because they are too large and takes memory
        # since this evaluator stores outputs of the entire dataset
        rles = [
            mask_util.encode(np.array(mask[:, :, None], order="F", dtype="uint8"))[0]
            for mask in instances.pred_masks
        ]
        for rle in rles:
            # "counts" is an array encoded by mask_util as a byte-stream. Python3's
            # json writer which always produces strings cannot serialize a bytestream
            # unless you decode it. Thankfully, utf-8 works out (which is also what
            # the pycocotools/_mask.pyx does).
            rle["counts"] = rle["counts"].decode("utf-8")

    has_keypoints = instances.has("pred_keypoints")
    if has_keypoints:
        keypoints = instances.pred_keypoints

    results = []
    for k in range(num_instance):
        if classes[k] != 0 or scores[k] < thresdhold:
            continue
        result = {
            "image_id": img_id,
            "category_id": classes[k],
            "bbox": boxes[k],
            "score": scores[k],
        }
        if has_mask:
            result["segmentation"] = rles[k]
        if has_keypoints:
            # In COCO annotations,
            # keypoints coordinates are pixel indices.
            # However our predictions are floating point coordinates.
            # Therefore we subtract 0.5 to be consistent with the annotation format.
            # This is the inverse of data loading logic in `datasets/coco.py`.
            keypoints[k][:, :2] -= 0.5
            result["keypoints"] = keypoints[k].flatten().tolist()
        results.append(result)
    return results


def instances_to_ctc(instances, img_id, file, thresdhold=0.01):
    """
    Dump an "Instances" object to a CTC format that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    scores = instances.scores
    classes = instances.pred_classes.tolist()

    masks = instances.pred_masks
    
    #根据scores对mask进行排序，对于重叠的mask，认为score的高的遮挡score低的mask
    _, index  = torch.sort(scores)

    seg = torch.zeros(instances.image_size)
    
    for i in index:
        if classes[i] == 0 and scores[i] > thresdhold:
            mask = masks[i]
            a = 1 +i.item()
            seg[mask > 0] = a 

    file_name = file.split("/")
    results = {}
    results["image_id"] = img_id
    results["segmentation"] = seg
    results["index_dataset"] = file_name[3]
    return results

def instances_to_ctc_test(instances, img_id, file, thresdhold = 0.01):
    """
    Dump an "Instances" object to a CTC format that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        return []

    scores = instances.scores
    classes = instances.pred_classes.tolist()

    masks = instances.pred_masks
    
    #根据scores对mask进行排序，对于重叠的mask，认为score的高的遮挡score低的mask
    _, index  = torch.sort(scores)

    seg = torch.zeros(instances.image_size)
    
    for i in index:
        if classes[i] == 0 and scores[i] > thresdhold:
            mask = masks[i]
            a = 1 +i.item()
            seg[mask > 0] = a 

    file_name = file.split("/")
    results = {}
    results["image_id"] = img_id
    results["segmentation"] = seg
    results["index_dataset"] = file_name[2]
    return results


def _evaluate_box_proposals(dataset_predictions, coco_api, thresholds=None, area="all", limit=None):
    """
    Evaluate detection proposal recall metrics. This function is a much
    faster alternative to the official COCO API recall evaluation code. However,
    it produces slightly different results.
    """
    # Record max overlap value for each gt box
    # Return vector of overlap values
    areas = {
        "all": 0,
        "small": 1,
        "medium": 2,
        "large": 3,
        "96-128": 4,
        "128-256": 5,
        "256-512": 6,
        "512-inf": 7,
    }
    area_ranges = [
        [0**2, 1e5**2],  # all
        [0**2, 32**2],  # small
        [32**2, 96**2],  # medium
        [96**2, 1e5**2],  # large
        [96**2, 128**2],  # 96-128
        [128**2, 256**2],  # 128-256
        [256**2, 512**2],  # 256-512
        [512**2, 1e5**2],
    ]  # 512-inf
    assert area in areas, "Unknown area range: {}".format(area)
    area_range = area_ranges[areas[area]]
    gt_overlaps = []
    num_pos = 0

    for prediction_dict in dataset_predictions:
        predictions = prediction_dict["boxes"]

        # sort predictions in descending order
        # TODO maybe remove this and make it explicit in the documentation
        inds = prediction_dict["objectness_logits"].sort(descending=True)[1]
        predictions = predictions[inds]

        ann_ids = coco_api.getAnnIds(imgIds=prediction_dict["image_id"])
        anno = coco_api.loadAnns(ann_ids)
        gt_boxes = [
            obj["bbox"]
            for obj in anno
            if obj["iscrowd"] == 0
        ]
        gt_boxes = torch.as_tensor(gt_boxes).reshape(-1, 5)  # guard against no boxes
        gt_boxes = RotatedBoxes(gt_boxes)
        gt_areas = torch.as_tensor([obj["area"] for obj in anno if obj["iscrowd"] == 0])

        if len(gt_boxes) == 0 or len(predictions) == 0:
            continue

        valid_gt_inds = (gt_areas >= area_range[0]) & (gt_areas <= area_range[1])
        gt_boxes = gt_boxes[valid_gt_inds]

        num_pos += len(gt_boxes)

        if len(gt_boxes) == 0:
            continue

        if limit is not None and len(predictions) > limit:
            predictions = predictions[:limit]

        overlaps = box_iou_rotated(predictions.float(), gt_boxes.tensor.float())

        _gt_overlaps = torch.zeros(len(gt_boxes))
        for j in range(min(len(predictions), len(gt_boxes))):
            # find which proposal box maximally covers each gt box
            # and get the iou amount of coverage for each gt box
            max_overlaps, argmax_overlaps = overlaps.max(dim=0)

            # find which gt box is 'best' covered (i.e. 'best' = most iou)
            gt_ovr, gt_ind = max_overlaps.max(dim=0)
            assert gt_ovr >= 0
            # find the proposal box that covers the best covered gt box
            box_ind = argmax_overlaps[gt_ind]
            # record the iou coverage of this gt box
            _gt_overlaps[j] = overlaps[box_ind, gt_ind]
            assert _gt_overlaps[j] == gt_ovr
            # mark the proposal box and the gt box as used
            overlaps[box_ind, :] = -1
            overlaps[:, gt_ind] = -1

        # append recorded iou coverage level
        gt_overlaps.append(_gt_overlaps)
    gt_overlaps = (
        torch.cat(gt_overlaps, dim=0) if len(gt_overlaps) else torch.zeros(0, dtype=torch.float32)
    )
    gt_overlaps, _ = torch.sort(gt_overlaps)

    if thresholds is None:
        step = 0.05
        thresholds = torch.arange(0.5, 0.95 + 1e-5, step, dtype=torch.float32)
    recalls = torch.zeros_like(thresholds)
    # compute recall for each iou threshold
    for i, t in enumerate(thresholds):
        recalls[i] = (gt_overlaps >= t).float().sum() / float(num_pos)
    # ar = 2 * np.trapz(recalls, thresholds)
    ar = recalls.mean()
    return {
        "ar": ar,
        "recalls": recalls,
        "thresholds": thresholds,
        "gt_overlaps": gt_overlaps,
        "num_pos": num_pos,
    }