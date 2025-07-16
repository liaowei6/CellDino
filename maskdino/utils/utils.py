import torch
import copy
from torch import nn, Tensor
import os
from typing import Any, Iterator, List, Union, Tuple
import math
import torch.nn.functional as F
import torchvision.ops as ops
from torch import nn
import numpy as np
import cv2
from .box_ops import box_cxcywh_to_xyxy
class BitMasks_ct:
    """
    This class stores the segmentation masks for all objects in one image, in
    the form of bitmaps.

    Attributes:
        tensor: bool Tensor of N,H,W, representing N instances in the image.
    """

    def __init__(self, tensor: Union[torch.Tensor, np.ndarray]):
        """
        Args:
            tensor: bool Tensor of N,H,W, representing N instances in the image.
        """
        if isinstance(tensor, torch.Tensor):
            tensor = tensor.to(torch.bool)
        else:
            tensor = torch.as_tensor(tensor, dtype=torch.bool, device=torch.device("cpu"))
        assert tensor.dim() == 3, tensor.size()
        self.image_size = tensor.shape[1:]
        self.tensor = tensor

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "BitMasks_ct":
        return BitMasks_ct(self.tensor.to(*args, **kwargs))

    @property
    def device(self) -> torch.device:
        return self.tensor.device

    @torch.jit.unused
    def __getitem__(self, item: Union[int, slice, torch.BoolTensor]) -> "BitMasks_ct":
        """
        Returns:
            BitMasks: Create a new :class:`BitMasks` by indexing.

        The following usage are allowed:

        1. `new_masks = masks[3]`: return a `BitMasks` which contains only one mask.
        2. `new_masks = masks[2:10]`: return a slice of masks.
        3. `new_masks = masks[vector]`, where vector is a torch.BoolTensor
           with `length = len(masks)`. Nonzero elements in the vector will be selected.

        Note that the returned object might share storage with this object,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return BitMasks_ct(self.tensor[item].unsqueeze(0))
        m = self.tensor[item]
        assert m.dim() == 3, "Indexing on BitMasks with {} returns a tensor with shape {}!".format(
            item, m.shape
        )
        return BitMasks_ct(m)

    @torch.jit.unused
    def __iter__(self) -> torch.Tensor:
        yield from self.tensor

    @torch.jit.unused
    def __repr__(self) -> str:
        s = self.__class__.__name__ + "("
        s += "num_instances={})".format(len(self.tensor))
        return s

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def nonempty(self) -> torch.Tensor:
        """
        Find masks that are non-empty.

        Returns:
            Tensor: a BoolTensor which represents
                whether each mask is empty (False) or non-empty (True).
        """
        return self.tensor.flatten(1).any(dim=1)


    def get_oriented_bounding_boxes(self, norm=True) -> torch.Tensor:
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)
        rectangles = []
        mask_np = self.tensor.cpu().numpy().astype(np.uint8)
        for i in range(mask_np.shape[0]):
            # 将当前mask转换为二值图像
            binary = (mask_np[i] > 0).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            kernel[0,0] = 0
            kernel[0,2] = 0
            kernel[2,0] = 0
            kernel[2,2] = 0
            binary_open =  cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            # 查找轮廓
            contours, _ = cv2.findContours(binary_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                #对于有多个外接矩形的mask取面积最大的哪一个作为斜方框
                area_max = 0
                for contour in contours:
                    r = cv2.minAreaRect(contour.astype(np.float32))
                    area = r[1][0] * r[1][1]
                    if area > area_max:
                        if norm:
                            rect = [r[0][0], r[0][1], r[1][1], r[1][0], (90 - r[2]) / 90.0]  #(x, y, w, h, theta)
                        else:
                            rect = [r[0][0], r[0][1], r[1][1], r[1][0], (90 - r[2])]
                        area_max = area
                #对于SIM+数据集，如果多个不连接的掩码则取包含所有的外接矩形
                # contour = contours[0]
                # for i in range(1, len(contours)):
                #     contour = np.concatenate((contour, contours[i]), axis=0)
                # r = cv2.minAreaRect(contour.astype(np.float32))
                # if norm:
                #     rect = [r[0][0], r[0][1], r[1][1], r[1][0], (90 - r[2]) / 90.0]  #(x, y, w, h, theta)
                # else:
                #     rect = [r[0][0], r[0][1], r[1][1], r[1][0], (90 - r[2])]
                # 获取最小外接矩形
                # contours = contours[0].astype(np.float32)
                # r = cv2.minAreaRect(contours)
                # if r[2] == 90:
                #     rect = [r[0][0], r[0][1], r[1][1], r[1][0], 0]
                # else:
                #     rect = [r[0][0], r[0][1], r[1][0], r[1][1], r[2] / 90.0]  #(x, y, w, h, theta)
            else:
                rect = [0.0, 0.0, 0.0, 0.0, 0.0]
            rectangles.append(rect)
        rectangles = np.array(rectangles)
        rectangles = torch.from_numpy(rectangles)
        return rectangles

    def get_oriented_bounding_boxes_crop(self, mask_center, norm=True) -> torch.Tensor:
        """
        Returns:
            Boxes: tight bounding boxes around bitmasks.
            If a mask is empty, it's bounding box will be all zero.
        """
        boxes = torch.zeros(self.tensor.shape[0], 4, dtype=torch.float32)
        rectangles = []
        mask_np = self.tensor.cpu().numpy().astype(np.uint8)
        h, w = mask_np.shape[-2:]
        mask_center = mask_center.cpu().numpy()
        for i in range(mask_np.shape[0]):
            # 将当前mask转换为二值图像
            binary = (mask_np[i] > 0).astype(np.uint8)
            kernel = np.ones((3, 3), np.uint8)
            kernel[0,0] = 0
            kernel[0,2] = 0
            kernel[2,0] = 0
            kernel[2,2] = 0
            binary_open =  cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            # 查找轮廓
            contours, _ = cv2.findContours(binary_open, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) > 0:
                #对于有多个外接矩形的mask取面积最大的哪一个作为斜方框
                area_max = 0
                for contour in contours:
                    r = cv2.minAreaRect(contour.astype(np.float32))
                    area = r[1][0] * r[1][1]
                    if area > area_max:
                        if norm:
                            rect = [r[0][0], r[0][1], r[1][1], r[1][0], (90 - r[2]) / 90.0]  #(x, y, w, h, theta)
                        else:
                            rect = [r[0][0], r[0][1], r[1][1], r[1][0], (90 - r[2])]
                        area_max = area
                # 获取最小外接矩形
                # contours = contours[0].astype(np.float32)
                # r = cv2.minAreaRect(contours)
                # if r[2] == 90:
                #     rect = [r[0][0], r[0][1], r[1][1], r[1][0], 0]
                # else:
                #     rect = [r[0][0], r[0][1], r[1][0], r[1][1], r[2] / 90.0]  #(x, y, w, h, theta)
            else:
                rect = [0.0, 0.0, 0.0, 0.0, 0.0]
            rect[:2] = rect[:2] + mask_center[i] - [w/2, h/2]
            rectangles.append(rect)
        rectangles = np.array(rectangles)
        rectangles = torch.from_numpy(rectangles)
        return rectangles
    @staticmethod
    def cat(bitmasks_list: List["BitMasks_ct"]) -> "BitMasks_ct":
        """
        Concatenates a list of BitMasks into a single BitMasks

        Arguments:
            bitmasks_list (list[BitMasks])

        Returns:
            BitMasks: the concatenated BitMasks
        """
        assert isinstance(bitmasks_list, (list, tuple))
        assert len(bitmasks_list) > 0
        assert all(isinstance(bitmask, BitMasks_ct) for bitmask in bitmasks_list)

        cat_bitmasks = type(bitmasks_list[0])(torch.cat([bm.tensor for bm in bitmasks_list], dim=0))
        return cat_bitmasks

class EBBoxes:
    """
    This structure stores a list of boxes as a Nx5 torch.Tensor.
    It supports some common methods about boxes
    (`area`, `clip`, `nonempty`, etc),
    and also behaves like a Tensor
    (support indexing, `to(device)`, `.device`, and iteration over all boxes)

    Attributes:
        tensor (torch.Tensor): float matrix of Nx8. Each row is (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4).
    """

    def __init__(self, tensor: torch.Tensor):
        """
        Args:
            tensor (Tensor[float]): a Nx8 matrix.  Each row is (x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4).
        """
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor, dtype=torch.float32, device=torch.device("cpu"))
        else:
            tensor = tensor.to(torch.float32)
        if tensor.numel() == 0:
            # Use reshape, so we don't end up creating a new tensor that does not depend on
            # the inputs (and consequently confuses jit)
            tensor = tensor.reshape((-1, 4)).to(dtype=torch.float32)
        assert tensor.dim() == 2 and tensor.size(-1) == 8, tensor.size()

        self.tensor = tensor
        self.tensor_5 = self.obox_xyxy_to_cxcywht(self.tensor)  #(cx, cy, w, h, theta)
    def obox_xyxy_to_cxcywht(self, x):  
        x_1, y_1, x_2, y_2, x_3, y_3, x_4, y_4 = x.unbind(-1)    
        b = [(x_1 + x_2 +x_3 +x_4) / 4, (y_1 + y_2 + y_3 + y_4) / 4,
            torch.sqrt(torch.square(x_2 - x_3) + torch.square(y_2 - y_3)), 
            torch.sqrt(torch.square(x_2 - x_1) + torch.square(y_2 - y_1)),
            torch.atan((x_2 - x_3) / (y_3 - y_2))]
        return torch.stack(b, dim=-1)

    def clone(self) -> "EBBoxes":
        """
        Clone the EBBoxes.

        Returns:
            Boxes
        """
        return EBBoxes(self.tensor.clone())

    def to(self, device: torch.device):
        # Boxes are assumed float32 and does not support to(dtype)
        return EBBoxes(self.tensor.to(device=device))

    def area(self) -> torch.Tensor:
        """
        Computes the area of all the boxes.

        Returns:
            torch.Tensor: a vector with areas of each box.
        """
        box = self.tensor_5
        area = (box[:,2] * box[:,3])
        return area

    def clip(self, box_size: Tuple[int, int]) -> None:
        """
        Clip (in place) the boxes by limiting x coordinates to the range [0, width]
        and y coordinates to the range [0, height].

        Args:
            box_size (height, width): The clipping box's size.
        """
        assert torch.isfinite(self.tensor).all(), "Box tensor contains infinite or NaN!"
        h, w = box_size
        x1 = self.tensor[:, 0].clamp(min=0, max=w)
        y1 = self.tensor[:, 1].clamp(min=0, max=h)
        x2 = self.tensor[:, 2].clamp(min=0, max=w)
        y2 = self.tensor[:, 3].clamp(min=0, max=h)
        x3 = self.tensor[:, 4].clamp(min=0, max=w)
        y3 = self.tensor[:, 5].clamp(min=0, max=h)
        x4 = self.tensor[:, 6].clamp(min=0, max=w)
        y4 = self.tensor[:, 7].clamp(min=0, max=h)
        self.tensor = torch.stack((x1, y1, x2, y2, x3, y3, x4, y4), dim=-1)

    def nonempty(self, threshold: float = 0.0) -> torch.Tensor:
        """
        Find boxes that are non-empty.
        A box is considered empty, if either of its side is no larger than threshold.

        Returns:
            Tensor:
                a binary vector which represents whether each box is empty
                (False) or non-empty (True).
        """
        box = self.tensor_5
        keep = (box[:, 2] > threshold) & (box[:, 3] > threshold)
        return keep

    def __getitem__(self, item) -> "EBBoxes":
        """
        Args:
            item: int, slice, or a BoolTensor

        Returns:
            Boxes: Create a new :class:`Boxes` by indexing.

        The following usage are allowed:

        1. `new_boxes = boxes[3]`: return a `Boxes` which contains only one box.
        2. `new_boxes = boxes[2:10]`: return a slice of boxes.
        3. `new_boxes = boxes[vector]`, where vector is a torch.BoolTensor
           with `length = len(boxes)`. Nonzero elements in the vector will be selected.

        Note that the returned Boxes might share storage with this Boxes,
        subject to Pytorch's indexing semantics.
        """
        if isinstance(item, int):
            return EBBoxes(self.tensor[item].view(1, -1))
        b = self.tensor[item]
        assert b.dim() == 2, "Indexing on Boxes with {} failed to return a matrix!".format(item)
        return EBBoxes(b)

    def __len__(self) -> int:
        return self.tensor.shape[0]

    def __repr__(self) -> str:
        return "EBBoxes(" + str(self.tensor) + ")"

    def inside_box(self, box_size: Tuple[int, int], boundary_threshold: int = 0) -> torch.Tensor:
        """
        Args:
            box_size (height, width): Size of the reference box.
            boundary_threshold (int): Boxes that extend beyond the reference box
                boundary by more than boundary_threshold are considered "outside".

        Returns:
            a binary vector, indicating whether each box is inside the reference box.
        """
        height, width = box_size
        inds_inside = (
            (self.tensor[..., 0] >= -boundary_threshold)
            & (self.tensor[..., 1] >= -boundary_threshold)
            & (self.tensor[..., 2] < width + boundary_threshold)
            & (self.tensor[..., 3] < height + boundary_threshold)
        )
        return inds_inside

    def get_centers(self) -> torch.Tensor:
        """
        Returns:
            The box centers in a Nx2 array of (x, y).
        """
        return self.tensor_5[:, 0:2]

    def scale(self, scale_x: float, scale_y: float) -> None:
        """
        Scale the box with horizontal and vertical scaling factors
        """
        self.tensor[:, 0::2] *= scale_x
        self.tensor[:, 1::2] *= scale_y

    @classmethod
    def cat(cls, boxes_list: List["EBBoxes"]) -> "EBBoxes":
        """
        Concatenates a list of Boxes into a single Boxes

        Arguments:
            boxes_list (list[Boxes])

        Returns:
            Boxes: the concatenated Boxes
        """
        assert isinstance(boxes_list, (list, tuple))
        if len(boxes_list) == 0:
            return cls(torch.empty(0))
        assert all([isinstance(box, EBBoxes) for box in boxes_list])

        # use torch.cat (v.s. layers.cat) so the returned boxes never share storage with input
        cat_boxes = cls(torch.cat([b.tensor for b in boxes_list], dim=0))
        return cat_boxes

    @property
    def device(self):
        return self.tensor.device

    # type "Iterator[torch.Tensor]", yield, and iter() not supported by torchscript
    # https://github.com/pytorch/pytorch/issues/18627
    @torch.jit.unused
    def __iter__(self):
        """
        Yield a box as a Tensor of shape (8,) at a time.
        """
        yield from self.tensor



class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)


def gen_encoder_output_proposals(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals

def gen_encoder_output_proposals_ct(memory:Tensor, memory_padding_mask:Tensor, spatial_shapes:Tensor):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 5
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur:(_cur + H_ * W_)].view(N_, H_, W_, 1)
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)

        grid_y, grid_x = torch.meshgrid(torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
                                        torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device))
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(N_, 1, 1, 2)
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale
        wh = torch.ones_like(grid) * 0.05 * (2.0 ** lvl)   #原来的
        # wh = torch.ones_like(grid) * 0.01 * (2.0 ** lvl)   #针对psc数据集
        # wh = torch.ones_like(grid) * 0.1 * (2.0 ** lvl)  #针对msc数据集
        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        #TODO 初始化角度时均匀初始化，而不是全为一个值
        theta = torch.full([N_, proposal.shape[1], 1], 0.5, device=proposal.device)
        proposal = torch.cat((proposal, theta), -1)
        proposals.append(proposal)
        _cur += (H_ * W_)
    output_proposals = torch.cat(proposals, 1)
    #由于角度初始化为全0，所以会导致valid为全false，所以将角度初始化为45度
    output_proposals_valid = ((output_proposals > 0.01) & (output_proposals < 0.99)).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))
    output_proposals = output_proposals.masked_fill(memory_padding_mask.unsqueeze(-1), float('inf'))
    output_proposals = output_proposals.masked_fill(~output_proposals_valid, float('inf'))

    output_memory = memory
    output_memory = output_memory.masked_fill(memory_padding_mask.unsqueeze(-1), float(0))
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))
    return output_memory, output_proposals, output_proposals_valid


def gen_sineembed_for_position(pos_tensor, temperature=10000):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    elif pos_tensor.size(-1) == 5:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)

        theta_embed = pos_tensor[:, :, 4] * scale
        pos_theta = theta_embed[:, :, None] / dim_t
        pos_theta = torch.stack((pos_theta[:, :, 0::2].sin(), pos_theta[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h, pos_theta), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def _get_clones(module, N, layer_share=False):

    if layer_share:
        return nn.ModuleList([module for i in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_rotated_rect_vertices(obboxes):
    # obboxes  (x,y,w,h, \theta) 全部都是归一化后的结果
    obboxes = obboxes.transpose(0,1)
    obboxes[:, :, 4] = obboxes[: ,:, 4] * math.pi / 2
    x = obboxes[:, :, 0]
    y = obboxes[:, :, 1]
    w = obboxes[:, :, 2]
    h = obboxes[:, :, 3]
    a = - obboxes[:, :, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    point_x = torch.stack([p1x, p2x, p3x, p4x], dim=2)
    point_y = torch.stack([p1y, p2y, p3y, p4y], dim=2)
    min_x = torch.min(point_x, dim=2).values  
    max_x = torch.max(point_x, dim=2).values
    min_y = torch.min(point_y, dim=2).values
    max_y = torch.max(point_y, dim=2).values
    widths = max_x - min_x
    heights = max_y - min_y
    return torch.max(widths), torch.max(heights)

def convert_boxes_to_roi_format(boxes):
    """
    将形状为 [B, N, 4] 的边界框转换为 ROIAlign 所需的 [R, 5] 格式
    :param boxes: 边界框，形状为 [B, N, 4]
    :param batch_size: 批量大小 B
    :return: 转换后的边界框，形状为 [R, 5]
    """
    B, N, _ = boxes.shape
    R = B * N

    # 展平边界框
    boxes_flat = boxes.view(R, 4)

    # 创建批次索引
    batch_indices = torch.arange(B, dtype=torch.float32, device=boxes.device).view(B, 1).expand(B, N).contiguous().view(R, 1)

    # 将批次索引添加到边界框前面
    rois = torch.cat([batch_indices, boxes_flat], dim=1)

    return rois

def extract_region_features(feature_map, centers, w, h):
    """
    从特征图中提取以一组中心点为中心、宽度为 w、高度为 h 的区域特征，填充超出边界的区域
    :param feature_map: 输入特征图 (B, C, H, W)
    :param centers: 中心点坐标 (N, B, 2)
    :param w: 区域宽度
    :param h: 区域高度
    :return: 提取的区域特征 (N, C, h, w)
    """
    B, C, H, W = feature_map.shape
    centers = centers.transpose(0, 1)
    _, N, _ = centers.shape

    # 将归一化的中心点和宽高转换为特征图上的实际坐标
    centers_actual = centers * torch.tensor([W, H], device=centers.device).view(1, 1, 2)
    w_actual = int(w * W) + 1
    h_actual = int(h * H) + 1

    # 计算填充大小
    pad_left = max(0, int(-torch.min(centers_actual[..., 0] - w_actual // 2))+1)
    pad_right = max(0, int(torch.max(centers_actual[..., 0] + w_actual // 2 - W))+1)
    pad_top = max(0, int(-torch.min(centers_actual[..., 1] - h_actual // 2))+1)
    pad_bottom = max(0, int(torch.max(centers_actual[..., 1] + h_actual // 2 - H))+1)

    # 对特征图进行填充
    feature_map_padded = F.pad(feature_map, (pad_left, pad_right, pad_top, pad_bottom), mode='constant', value=0)

    # 调整中心点坐标
    centers_padded = centers_actual + torch.tensor([pad_left, pad_top], device=centers_actual.device).view(1, 1, 2)
    # 转换为边框
    crop_boxes = torch.cat([centers_padded, torch.tensor([w_actual, h_actual], device=centers_padded.device).view(1, 1, 2).expand(B, N, 2)], dim=-1)
    crop_boxes = box_cxcywh_to_xyxy(crop_boxes)
    crop_boxes = convert_boxes_to_roi_format(crop_boxes)
    #采用ROI_ALIGN获取区域特征
    roi_features = ops.roi_align(feature_map_padded, crop_boxes.to(torch.float32), (h_actual, w_actual))
    return roi_features.view(B, N, C, int(h_actual), int(w_actual))