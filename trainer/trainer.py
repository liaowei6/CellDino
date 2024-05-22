from detectron2.engine.defaults import DefaultTrainer
import logging
import numpy as np
import time
import torch
from torch.nn.parallel import DataParallel, DistributedDataParallel
from maskdino.cgtdino import Tracker
import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage
import logging
from collections import OrderedDict
from typing import Optional
import torch
from fvcore.nn.precise_bn import get_bn_modules
from omegaconf import OmegaConf
from torch.nn.parallel import DistributedDataParallel

import detectron2.data.transforms as T
from detectron2.evaluation import (
    DatasetEvaluator,
    print_csv_format,
)
from ..evaluation.inference import inference_on_dataset,inference_on_dataset_submit,inference_on_dataset_cell,inference_on_dataset_track
from detectron2.utils import comm
from detectron2.engine.train_loop import AMPTrainer
from detectron2.data import build_detection_test_loader

class DefaultTrainer_cell(DefaultTrainer):
    """
    A trainer for cell 
    """
    def __init__(self):
        super(DefaultTrainer, self).__init__()

    @classmethod
    def test(cls, cfg, model, evaluators=None, mapper=None, cell=False, track=False): #add mapper
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
            cell: if livecell dataset
            track: track

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = cls.build_test_loader(cfg, dataset_name, mapper)
            # import torch.multiprocessing
            # torch.multiprocessing.set_sharing_strategy('file_system')
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = cls.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            if cfg.SUBMIT:
                results_i = inference_on_dataset_submit(model, data_loader, evaluator, submit_dir = cfg.SUBMIT_DIR)
            elif cell:
                results_i = inference_on_dataset_cell(model, data_loader, evaluator)
            elif cfg.TRACK:
                results_i = inference_on_dataset_track(model, data_loader, evaluator)
            else:
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results

    @classmethod
    def build_test_loader(cls, cfg, dataset_name, mapper=None):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

class AMPTrainer_cell(AMPTrainer):
    """
    Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precision
    in the training loop.
    """

    def __init__(
        self,
        model,
        data_loader,
        optimizer,
        gather_metric_period=1,
        zero_grad_before_forward=False,
        grad_scaler=None,
        precision: torch.dtype = torch.float16,
        log_grad_scaler: bool = False,
        async_write_metrics=False,
    ):
        """
        Args:
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward,
                async_write_metrics: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
            precision: torch.dtype as the target precision to cast to in computations
        """
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward
        )

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler

            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler

    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        # 过滤掉实例数量为0的样本和大于100的
        num = 1
        for i in data:
            num = num * len(i["instances"])
        while not num:
            data = next(self._data_loader_iter)
            num = 1
            for i in data:
                num = num * len(i["instances"])
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward and self.iter % self.accumulation_steps == 0:
            self.optimizer.zero_grad()
        with autocast():
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())
        print("total_loss:{},loss_ce:{},loss_mask:{},loss_dice:{},loss_giou:{},loss_bbox:{}".format(losses, loss_dict["loss_ce"], loss_dict["loss_mask"], loss_dict["loss_dice"], loss_dict["loss_giou"], loss_dict["loss_bbox"]))
        if not self.zero_grad_before_forward and self.iter % self.accumulation_steps == 0:
            self.optimizer.zero_grad()
        losses = losses / self.accumulation_steps
        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics:
            # write metrics asynchronically
            self.concurrent_executor.submit(
                self._write_metrics, loss_dict, data_time, iter=self.iter
            )
        else:
            self._write_metrics(loss_dict, data_time)
        if (self.iter +1) % self.accumulation_steps == 0:
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])

class AMPTrackTrainer_cell(AMPTrainer):
    """
    For training of cell track
    """


    def run_step(self):
        """
        Implement the AMP training logic.
        """
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)  #每个batch包含两张连续的图片 (batch_size, 2, )
        # 过滤掉实例数量为0的样本和大于100的
        data_second = []
        num = 1
        for i in data:
            num = num * len(i["instances"])
            data_second.append(i["second"])
            i.pop("second")
        while not num:
            data = next(self._data_loader_iter)
            num = 1
            for i in data:
                num = num * len(i["instances"])
        data_time = time.perf_counter() - start
        data_set = [data, data_second]
        batch_size = len(data)
        del data
        del data_second
        Cell_Tracker = Tracker()
        Cell_Tracker.reset()
        for i in range(2):
            if self.zero_grad_before_forward:
                self.optimizer.zero_grad()
            
            with autocast():
                loss_dict = self.model(data_set[i], tracker = Cell_Tracker)
                if isinstance(loss_dict, torch.Tensor):
                    losses = loss_dict
                    loss_dict = {"total_loss": loss_dict}
                else:
                    losses = sum(loss_dict.values())
            print("total_loss:{},loss_ce:{},loss_mask:{},loss_dice:{},loss_giou:{},loss_bbox:{}".format(losses, loss_dict["loss_ce"], loss_dict["loss_mask"], loss_dict["loss_dice"], loss_dict["loss_giou"], loss_dict["loss_bbox"]))
            if not self.zero_grad_before_forward:
                self.optimizer.zero_grad()

            self.grad_scaler.scale(losses).backward()

            if self.log_grad_scaler:
                storage = get_event_storage()
                storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

            self.after_backward()

            if self.async_write_metrics:
                # write metrics asynchronically
                self.concurrent_executor.submit(
                    self._write_metrics, loss_dict, data_time, iter=self.iter
                )
            else:
                self._write_metrics(loss_dict, data_time)

            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
        