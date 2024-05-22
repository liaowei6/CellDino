import datetime
import logging
import time
from collections import OrderedDict, abc
from contextlib import ExitStack, contextmanager
from typing import List, Union
import torch
from torch import nn
from tifffile import imsave
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from .visualizer import Visualizer_cell
#from maskdino.tracker import Tracker
from collections import deque
import os
import numpy as np
from detectron2.evaluation.evaluator import DatasetEvaluator, DatasetEvaluators, inference_context


def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            # image = inputs[0]['image']
            # image = image.permute(1, 2, 0)
            image_raw = inputs[0]['image_raw']
            image_raw = image_raw.permute(1, 2, 0)
            visualizer = Visualizer_cell(image_raw)
            visualizer_true = Visualizer_cell(image_raw)
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if len(outputs[0]["instances"]) == 0:
                return
            # 可视化
            #boxes= inputs[0]["bbox"].to(torch.device("cpu"))
            #boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            item_vis_output = visualizer.draw_instance_predictions(outputs[0]["instances"].to(torch.device("cpu")))
            # item_vis_output = visualizer_true.overlay_instances(boxes = boxes)
            # item_vis_output = visualizer_true.overlay_rotated_instances(inputs[0]["bbox"].to(torch.device("cpu")))
            item_vis_output.save("output/images/" + str(inputs[0]['image_id']) +".png")
            # if vis_output != None:
            #     vis_output.save("output/images/" + str(inputs[0]['image_id']) +".png")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def inference_on_dataset_track(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        tracker = Tracker()
        datasets_index_last = None
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            # image = inputs[0]['image']
            # image = image.permute(1, 2, 0)
            datasets_index = inputs[0]["file_name"].split("/")[-3]
            if datasets_index != datasets_index_last:
                datasets_index_last = datasets_index 
                tracker.reset()
            image_raw = inputs[0]['image_raw']
            image_raw = image_raw.permute(1, 2, 0)
            visualizer = Visualizer_cell(image_raw)
            visualizer_true = Visualizer_cell(image_raw)
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs, tracker=tracker)
            if len(outputs[0]["instances"]) == 0:
                return
            # 可视化
            #boxes= inputs[0]["bbox"].to(torch.device("cpu"))
            #boxes = BoxMode.convert(boxes, BoxMode.XYWH_ABS, BoxMode.XYXY_ABS)
            item_vis_output = visualizer.draw_instance_predictions(outputs[0]["instances"].to(torch.device("cpu")))
            # item_vis_output = visualizer_true.overlay_instances(boxes = boxes)
            # item_vis_output = visualizer_true.overlay_rotated_instances(inputs[0]["bbox"].to(torch.device("cpu")))
            item_vis_output.save("output/images/" + str(inputs[0]['image_id']) +".png")
            # if vis_output != None:
            #     vis_output.save("output/images/" + str(inputs[0]['image_id']) +".png")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results


def inference_on_dataset_cell(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None],
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            image = inputs[0]['image']
            image = image.permute(1, 2, 0)
            image_raw = inputs[0]['image_raw']
            image_raw = image_raw.permute(1, 2, 0)
            visualizer = Visualizer_cell(image_raw)
            # visualizer_item = Visualizer(image_raw)
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if len(outputs[0]["instances"]) == 0:
                return
            vis_output = visualizer.draw_instance_predictions(outputs[0]["instances"].to(torch.device("cpu")))
            vis_output.save("output/images/" + str(inputs[0]['image_id']) +".png")
            # item_vis_output = visualizer_item.draw_instance_predictions(outputs[0]["item_instances"].to(torch.device("cpu")),thresholds=0)
            # item_vis_output = visualizer_item.overlay_rotated_instances(outputs[0]["item_box"].to(torch.device("cpu")))
            # item_vis_output.save("output/images_item/" + str(inputs[0]['image_id']) +".png")
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process_test(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time
            
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    masks = []
    index_dataset = []
    for prediction in evaluator._predictions:
        masks.append(prediction["ctc"]["segmentation"])
        index_dataset.append(prediction["ctc"]["index_dataset"])
    result_path = "output/hela_test/"
    id = 0
    last_index = index_dataset[0]
    for index, mask in zip(index_dataset, masks):
        if index != last_index:
            last_index = index
            id = 0
        mask = mask.numpy()
        mask_file = os.path.join(result_path + index + "_RES", "mask" + str(id).zfill(3) + ".tif")
        id += 1
        imsave(mask_file, mask.astype(np.uint16),)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    results = {}
    return results

def inference_on_dataset_submit(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], submit_dir = '',
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if len(outputs[0]["instances"]) == 0:
                return
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process_test(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time
            
            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    masks = []
    index_dataset = []
    for prediction in evaluator._predictions:
        masks.append(prediction["ctc"]["segmentation"])
        index_dataset.append(prediction["ctc"]["index_dataset"])
    result_path = submit_dir
    id = 0
    for mask in masks:
        mask = mask.numpy()
        mask_file = os.path.join(result_path, "mask" + str(id).zfill(3) + ".tif")
        id += 1
        imsave(mask_file, mask.astype(np.uint16),)
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    results = {}
    return results

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
        track_ids = []
        for track in self.tracks:
            if track.has_positive_area and track.state != 2:
                tracks.append(track)
                track_pos.append(track.pos.unsqueeze(0))
                track_query.append(track.query_emb.unsqueeze(0))
                track_ids.append(track.id)
        self.track_ids = track_ids
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
        self.state = 3   #0代表inactive, 1代表track, 2表示lost 
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