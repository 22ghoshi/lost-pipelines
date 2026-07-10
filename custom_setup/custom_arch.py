#!/usr/bin/env python3

import math
import torch
import torch.nn as nn
import torchvision
import copy
import numpy as np
import cv2

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    """Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)."""

    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        """Initialize Conv layer with given arguments including activation."""
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        """Apply convolution, batch normalization and activation to input tensor."""
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        """Perform transposed convolution of 2D data."""
        return self.act(self.conv(x))
    
class Bottleneck(nn.Module):
    """Standard bottleneck."""

    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        """Initializes a standard bottleneck module with optional shortcut connection and configurable parameters."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        """Applies the YOLO FPN to input data."""
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    
class C2f(nn.Module):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        """Initializes a CSP bottleneck with 2 convolutions and n Bottleneck blocks for faster processing."""
        super().__init__()
        self.c = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3), (3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        """Forward pass through C2f layer."""
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

    def forward_split(self, x):
        """Forward pass using split() instead of chunk()."""
        y = list(self.cv1(x).split((self.c, self.c), 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))

class C3(nn.Module):
    """CSP Bottleneck with 3 convolutions."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        """Initialize the CSP Bottleneck with given channels, number, shortcut, groups, and expansion values."""
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=((1, 1), (3, 3)), e=1.0) for _ in range(n)))

    def forward(self, x):
        """Forward pass through the CSP bottleneck with 2 convolutions."""
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))

class C3k(C3):
    """C3k is a CSP bottleneck module with customizable kernel sizes for feature extraction in neural networks."""

    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5, k=3):
        """Initializes the C3k module with specified channels, number of layers, and configurations."""
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        # self.m = nn.Sequential(*(RepBottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, k=(k, k), e=1.0) for _ in range(n)))
    
class C3k2(C2f):
    """Faster Implementation of CSP Bottleneck with 2 convolutions."""

    def __init__(self, c1, c2, n=1, c3k=False, e=0.5, g=1, shortcut=True):
        """Initializes the C3k2 module, a faster CSP Bottleneck with 2 convolutions and optional C3k blocks."""
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(
            C3k(self.c, self.c, 2, shortcut, g) if c3k else Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))
        
class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher."""

    def __init__(self, c1, c2, k=5):
        """
        Initializes the SPPF layer with given input/output channels and kernel size.

        This module is equivalent to SPP(k=(5, 9, 13)).
        """
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        """Forward pass through Ghost Convolution block."""
        y = [self.cv1(x)]
        y.extend(self.m(y[-1]) for _ in range(3))
        return self.cv2(torch.cat(y, 1))

class Attention(nn.Module):
    """
    Attention module that performs self-attention on the input tensor.

    Args:
        dim (int): The input tensor dimension.
        num_heads (int): The number of attention heads.
        attn_ratio (float): The ratio of the attention key dimension to the head dimension.

    Attributes:
        num_heads (int): The number of attention heads.
        head_dim (int): The dimension of each attention head.
        key_dim (int): The dimension of the attention key.
        scale (float): The scaling factor for the attention scores.
        qkv (Conv): Convolutional layer for computing the query, key, and value.
        proj (Conv): Convolutional layer for projecting the attended values.
        pe (Conv): Convolutional layer for positional encoding.
    """

    def __init__(self, dim, num_heads=8, attn_ratio=0.5):
        """Initializes multi-head attention module with query, key, and value convolutions and positional encoding."""
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.key_dim = int(self.head_dim * attn_ratio)
        self.scale = self.key_dim**-0.5
        nh_kd = self.key_dim * num_heads
        h = dim + nh_kd * 2
        self.qkv = Conv(dim, h, 1, act=False)
        self.proj = Conv(dim, dim, 1, act=False)
        self.pe = Conv(dim, dim, 3, 1, g=dim, act=False)

    def forward(self, x):
        """
        Forward pass of the Attention module.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            (torch.Tensor): The output tensor after self-attention.
        """
        B, C, H, W = x.shape
        N = H * W
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, self.num_heads, self.key_dim * 2 + self.head_dim, N).split(
            [self.key_dim, self.key_dim, self.head_dim], dim=2
        )

        attn = (q.transpose(-2, -1) @ k) * self.scale
        attn = attn.softmax(dim=-1)
        x = (v @ attn.transpose(-2, -1)).view(B, C, H, W) + self.pe(v.reshape(B, C, H, W))
        x = self.proj(x)
        return x

class PSABlock(nn.Module):
    """
    This class encapsulates the functionality for applying multi-head attention and feed-forward neural network layers with optional shortcut connections.
    """

    def __init__(self, c, attn_ratio=0.5, num_heads=4, shortcut=True) -> None:
        """Initializes the PSABlock with attention and feed-forward layers for enhanced feature extraction."""
        super().__init__()

        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
        self.ffn = nn.Sequential(Conv(c, c * 2, 1), Conv(c * 2, c, 1, act=False))
        self.add = shortcut

    def forward(self, x):
        """Executes a forward pass through PSABlock, applying attention and feed-forward layers to the input tensor."""
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x
        
class C2PSA(nn.Module):
    """
    This module implements a convolutional block with attention mechanisms to enhance feature extraction and processing
    capabilities. It includes a series of PSABlock modules for self-attention and feed-forward operations.
    """

    def __init__(self, c1, c2, n=1, e=0.5):
        """Initializes the C2PSA module with specified input/output channels, number of layers, and expansion ratio."""
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1)

        self.m = nn.Sequential(*(PSABlock(self.c, attn_ratio=0.5, num_heads=self.c // 64) for _ in range(n)))

    def forward(self, x):
        """Processes the input tensor 'x' through a series of PSA blocks and returns the transformed tensor."""
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))

class Concat(nn.Module):
    """Concatenate a list of tensors along dimension."""

    def __init__(self, dimension=1):
        """Concatenates a list of tensors along a specified dimension."""
        super().__init__()
        self.d = dimension

    def forward(self, x):
        """Forward pass for the YOLOv8 mask Proto module."""
        return torch.cat(x, self.d)
    
def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat([c_xy, wh], dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox

def make_anchors(feats, strides, grid_cell_offset=0.5):
    """Generate anchors from features."""
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i in range(len(feats)):  # use len(feats) to avoid TracerWarning from iterating over strides tensor
        stride = strides[i]
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + grid_cell_offset  # shift x
        sy = torch.arange(end=h, device=device, dtype=dtype) + grid_cell_offset  # shift y
        # Use row-major (y, x) anchor ordering so anchors line up with the
        # flattened feature-map order produced by .view(bs, c, -1).
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)

class Detect(nn.Module):
    """YOLO Detect head for object detection models.

    This class implements the detection head used in YOLO models for predicting bounding boxes and class probabilities.
    It supports both training and inference modes, with optional end-to-end detection capabilities.

    Attributes:
        dynamic (bool): Force grid reconstruction.
        export (bool): Export mode flag.
        format (str): Export format.
        end2end (bool): End-to-end detection mode.
        max_det (int): Maximum detections per image.
        shape (tuple): Input shape.
        anchors (torch.Tensor): Anchor points.
        strides (torch.Tensor): Feature map strides.
        legacy (bool): Backward compatibility for v3/v5/v8/v9/v11 models.
        xyxy (bool): Output format, xyxy or xywh.
        nc (int): Number of classes.
        nl (int): Number of detection layers.
        reg_max (int): DFL channels.
        no (int): Number of outputs per anchor.
        stride (torch.Tensor): Strides computed during build.
        cv2 (nn.ModuleList): Convolution layers for box regression.
        cv3 (nn.ModuleList): Convolution layers for classification.
        dfl (nn.Module): Distribution Focal Loss layer.
        one2one_cv2 (nn.ModuleList): One-to-one convolution layers for box regression.
        one2one_cv3 (nn.ModuleList): One-to-one convolution layers for classification.

    Methods:
        forward: Perform forward pass and return predictions.
        bias_init: Initialize detection head biases.
        decode_bboxes: Decode bounding boxes from predictions.
        postprocess: Post-process model predictions.

    Examples:
        Create a detection head for 80 classes
        >>> detect = Detect(nc=80, ch=(256, 512, 1024))
        >>> x = [torch.randn(1, 256, 80, 80), torch.randn(1, 512, 40, 40), torch.randn(1, 1024, 20, 20)]
        >>> outputs = detect(x)
    """

    dynamic = False  # force grid reconstruction
    export = False  # export mode
    format = None  # export format
    max_det = 300  # max_det
    agnostic_nms = False
    shape = None
    anchors = torch.empty(0)  # init
    strides = torch.empty(0)  # init
    legacy = False  # backward compatibility for v3/v5/v8/v9 models
    xyxy = False  # xyxy or xywh output

    def __init__(self, nc: int = 80, reg_max=16, end2end=False, ch: tuple = ()):
        """Initialize the YOLO detection layer with specified number of classes and channels.

        Args:
            nc (int): Number of classes.
            reg_max (int): Maximum number of DFL channels.
            end2end (bool): Whether to use end-to-end NMS-free detection.
            ch (tuple): Tuple of channel sizes from backbone feature maps.
        """
        super().__init__()
        self.nc = nc  # number of classes
        self.nl = len(ch)  # number of detection layers
        self.reg_max = reg_max  # DFL channels (ch[0] // 16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max * 4  # number of outputs per anchor
        self.stride = torch.zeros(self.nl)  # strides computed during build
        c2, c3 = max((16, ch[0] // 4, self.reg_max * 4)), max(ch[0], min(self.nc, 100))  # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(Conv(x, c2, 3), Conv(c2, c2, 3), nn.Conv2d(c2, 4 * self.reg_max, 1)) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x, c3, 3), Conv(c3, c3, 3), nn.Conv2d(c3, self.nc, 1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x, x, 3), Conv(x, c3, 1)),
                    nn.Sequential(DWConv(c3, c3, 3), Conv(c3, c3, 1)),
                    nn.Conv2d(c3, self.nc, 1),
                )
                for x in ch
            )
        )
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        if end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)

    @property
    def one2many(self):
        """Returns the one-to-many head components, here for v3/v5/v8/v9/v11 backward compatibility."""
        return dict(box_head=self.cv2, cls_head=self.cv3)

    @property
    def one2one(self):
        """Returns the one-to-one head components."""
        return dict(box_head=self.one2one_cv2, cls_head=self.one2one_cv3)

    @property
    def end2end(self):
        """Checks if the model has one2one for v3/v5/v8/v9/v11 backward compatibility."""
        return getattr(self, "_end2end", True) and hasattr(self, "one2one")

    @end2end.setter
    def end2end(self, value):
        """Override the end-to-end detection mode."""
        self._end2end = value

    def forward_head(
        self, x: list[torch.Tensor], box_head: torch.nn.Module = None, cls_head: torch.nn.Module = None
    ) -> dict[str, torch.Tensor]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        if box_head is None or cls_head is None:  # for fused inference
            return dict()
        bs = x[0].shape[0]  # batch size
        boxes = torch.cat([box_head[i](x[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], dim=-1)
        scores = torch.cat([cls_head[i](x[i]).view(bs, self.nc, -1) for i in range(self.nl)], dim=-1)
        return dict(boxes=boxes, scores=scores, feats=x)

    def forward(
        self, x: list[torch.Tensor]
    ) -> dict[str, torch.Tensor] | torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Concatenates and returns predicted bounding boxes and class probabilities."""
        preds = self.forward_head(x, **self.one2many)
        if self.end2end:
            x_detach = [xi.detach() for xi in x]
            one2one = self.forward_head(x_detach, **self.one2one)
            preds = {"one2many": preds, "one2one": one2one}
        if self.training:
            return preds
        y = self._inference(preds["one2one"] if self.end2end else preds)
        if self.end2end:
            y = self.postprocess(y.permute(0, 2, 1))
        return y if self.export else (y, preds)

    def _inference(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps.

        Args:
            x (dict[str, torch.Tensor]): Dictionary of predictions from detection layers.

        Returns:
            (torch.Tensor): Concatenated tensor of decoded bounding boxes and class probabilities.
        """
        # Inference path
        dbox = self._get_decode_boxes(x)
        return torch.cat((dbox, x["scores"].sigmoid()), 1)

    def _get_decode_boxes(self, x: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get decoded boxes based on anchors and strides."""
        shape = x["feats"][0].shape  # BCHW
        if self.dynamic or self.shape != shape:
            self.anchors, self.strides = (a.transpose(0, 1) for a in make_anchors(x["feats"], self.stride, 0.5))
            self.shape = shape

        dbox = self.decode_bboxes(self.dfl(x["boxes"]), self.anchors.unsqueeze(0)) * self.strides
        return dbox

    def bias_init(self):
        """Initialize Detect() biases, WARNING: requires stride availability."""
        for i, (a, b) in enumerate(zip(self.one2many["box_head"], self.one2many["cls_head"])):  # from
            a[-1].bias.data[:] = 2.0  # box
            b[-1].bias.data[: self.nc] = math.log(
                5 / self.nc / (640 / self.stride[i]) ** 2
            )  # cls (.01 objects, 80 classes, 640 img)
        if self.end2end:
            for i, (a, b) in enumerate(zip(self.one2one["box_head"], self.one2one["cls_head"])):  # from
                a[-1].bias.data[:] = 2.0  # box
                b[-1].bias.data[: self.nc] = math.log(
                    5 / self.nc / (640 / self.stride[i]) ** 2
                )  # cls (.01 objects, 80 classes, 640 img)

    def decode_bboxes(self, bboxes: torch.Tensor, anchors: torch.Tensor, xywh: bool = True) -> torch.Tensor:
        """Decode bounding boxes from predictions."""
        return dist2bbox(
            bboxes,
            anchors,
            xywh=xywh and not self.end2end and not self.xyxy,
            dim=1,
        )

    def postprocess(self, preds: torch.Tensor) -> torch.Tensor:
        """Post-processes YOLO model predictions.

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size, num_anchors, 4 + nc) with last dimension
                format [x1, y1, x2, y2, class_probs].

        Returns:
            (torch.Tensor): Processed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x1, y1, x2, y2, max_class_prob, class_index].
        """
        boxes, scores = preds.split([4, self.nc], dim=-1)
        scores, conf, idx = self.get_topk_index(scores, self.max_det)
        boxes = boxes.gather(dim=1, index=idx.repeat(1, 1, 4))
        return torch.cat([boxes, scores, conf], dim=-1)

    def get_topk_index(self, scores: torch.Tensor, max_det: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get top-k indices from scores.

        Args:
            scores (torch.Tensor): Scores tensor with shape (batch_size, num_anchors, num_classes).
            max_det (int): Maximum detections per image.

        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): Top scores, class indices, and filtered indices.
        """
        batch_size, anchors, nc = scores.shape  # i.e. shape(16,8400,84)
        # Use max_det directly during export for TensorRT compatibility (requires k to be constant),
        # otherwise use min(max_det, anchors) for safety with small inputs during Python inference
        k = max_det if self.export else min(max_det, anchors)
        if self.agnostic_nms:
            scores, labels = scores.max(dim=-1, keepdim=True)
            scores, indices = scores.topk(k, dim=1)
            labels = labels.gather(1, indices)
            return scores, labels, indices
        ori_index = scores.max(dim=-1)[0].topk(k)[1].unsqueeze(-1)
        scores = scores.gather(dim=1, index=ori_index.repeat(1, 1, nc))
        scores, index = scores.flatten(1).topk(k)
        idx = ori_index[torch.arange(batch_size)[..., None], index // nc]  # original index
        return scores[..., None], (index % nc)[..., None].float(), idx

    def fuse(self) -> None:
        """Remove the one2many head for inference optimization."""
        self.cv2 = self.cv3 = None

class DFL(nn.Module):
    """Integral module of Distribution Focal Loss (DFL).

    Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    """

    def __init__(self, c1: int = 16):
        """Initialize a convolutional layer with a given number of input channels.

        Args:
            c1 (int): Number of input channels.
        """
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False)
        self.conv.requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float32)
        with torch.no_grad():
            self.conv.weight.copy_(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the DFL module to input tensor and return transformed output."""
        b, _, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)

class DWConv(Conv):
    """Depth-wise convolution module."""

    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):
        """Initialize depth-wise convolution with given parameters.

        Args:
            c1 (int): Number of input channels.
            c2 (int): Number of output channels.
            k (int): Kernel size.
            s (int): Stride.
            d (int): Dilation.
            act (bool | nn.Module): Activation function.
        """
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)

class YOLO11nBackbone(nn.Module):
    """
    YOLO11n backbone — width=0.25, depth=0.50 scaling of the YOLO11 reference.

    Channel progression
    -------------------
    stem       : 3  → 16   (stride  2)
    stage1     : 16 → 32   (stride  4)   C3k2 → 64
    stage2     : 64 → 64   (stride  8)   C3k2 → 128   P3
    stage3     : 128 → 128  (stride 16)  C3k2 → 128   P4
    stage4     : 128 → 256  (stride 32)  C3k2 → 256 → SPPF → C2PSA   P5
    """

    def __init__(self):
        super().__init__()

        # ── stem (stride 2 → 1/2)
        self.conv0 = Conv(3, 16, 3, 2)          # P1/2 – 16ch

        # ── stage 1 (stride 2 → 1/4)
        self.conv1  = Conv(16, 32, 3, 2)         # 1/4 – 32ch
        self.c3k2_2 = C3k2(32, 64, n=1, c3k=False, e=0.25)   # P2 – 64ch

        # ── stage 2 (stride 2 → 1/8)
        self.conv3  = Conv(64, 64, 3, 2)          # 1/8 – 64ch
        self.c3k2_4 = C3k2(64, 128, n=1, c3k=False, e=0.25)  # P3 – 128ch

        # ── stage 3 (stride 2 → 1/16)
        self.conv5  = Conv(128, 128, 3, 2)        # 1/16 – 128ch
        self.c3k2_6 = C3k2(128, 128, n=1, c3k=True)           # P4 – 128ch

        # ── stage 4 (stride 2 → 1/32)
        self.conv7   = Conv(128, 256, 3, 2)       # 1/32 – 256ch
        self.c3k2_8  = C3k2(256, 256, n=1, c3k=True)          # 256ch
        self.sppf    = SPPF(256, 256, k=5)                     # 256ch
        self.dropblock = torchvision.ops.DropBlock2d(p=0.05, block_size=3)
        self.c2psa   = C2PSA(256, 256, n=1)                    # P5 – 256ch

    def forward(self, x):
        """
        Returns
        -------
        p3 : (B, 128, H/8,  W/8)
        p4 : (B, 128, H/16, W/16)
        p5 : (B, 256, H/32, W/32)
        """
        x = self.conv0(x)

        x = self.conv1(x)
        x = self.c3k2_2(x)         # P2 – not used by detect head

        x = self.conv3(x)
        p3 = self.c3k2_4(x)        # P3 feature

        x = self.conv5(p3)
        p4 = self.c3k2_6(x)        # P4 feature

        x = self.conv7(p4)
        x = self.c3k2_8(x)
        x = self.sppf(x)
        x = self.dropblock(x)
        p5 = self.c2psa(x)         # P5 feature

        return p3, p4, p5


class YOLO11nNeck(nn.Module):
    """
    FPN + PAN neck matching the YOLO11n head layers 11-22.

    Input channel map
    -----------------
    p3_in : 128  (backbone P3, 1/8)
    p4_in : 128  (backbone P4, 1/16)
    p5_in : 256  (backbone P5, 1/32)

    Output channel map
    ------------------
    out_s : 64   (1/8  – small objects)
    out_m : 128  (1/16 – medium objects)
    out_l : 256  (1/32 – large objects)
    """

    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # ── top-down: P5 → P4
        # Concat(p5_up, p4_in) = 256+128 = 384ch
        self.c3k2_td1 = C3k2(384, 128, n=1, c3k=False)    # → 128ch

        # ── top-down: P4_td → P3
        # Concat(p4_td_up, p3_in) = 128+128 = 256ch
        self.c3k2_td2 = C3k2(256, 64, n=1, c3k=False)     # → 64ch  (out_s)

        # ── bottom-up: P3_out → P4
        self.conv_bu1 = Conv(64, 64, 3, 2)
        # Concat(p3_down, p4_td) = 64+128 = 192ch
        self.c3k2_bu1 = C3k2(192, 128, n=1, c3k=False)    # → 128ch (out_m)

        # ── bottom-up: P4_out → P5
        self.conv_bu2 = Conv(128, 128, 3, 2)
        # Concat(p4_down, p5_in) = 128+256 = 384ch
        self.c3k2_bu2 = C3k2(384, 256, n=1, c3k=True)     # → 256ch (out_l)

    def forward(self, p3, p4, p5):
        # ── FPN top-down
        p5_up   = self.upsample(p5)                        # 256ch, 1/16
        td1_in  = torch.cat([p5_up, p4], dim=1)           # 384ch
        p4_td   = self.c3k2_td1(td1_in)                   # 128ch, 1/16

        p4_up   = self.upsample(p4_td)                    # 128ch, 1/8
        td2_in  = torch.cat([p4_up, p3], dim=1)           # 256ch
        out_s   = self.c3k2_td2(td2_in)                   # 64ch,  1/8

        # ── PAN bottom-up
        p3_down = self.conv_bu1(out_s)                     # 64ch,  1/16
        bu1_in  = torch.cat([p3_down, p4_td], dim=1)      # 192ch
        out_m   = self.c3k2_bu1(bu1_in)                   # 128ch, 1/16

        p4_down = self.conv_bu2(out_m)                     # 128ch, 1/32
        bu2_in  = torch.cat([p4_down, p5], dim=1)         # 384ch
        out_l   = self.c3k2_bu2(bu2_in)                   # 256ch, 1/32

        return out_s, out_m, out_l


class YOLO11n(nn.Module):
    """
    YOLO11 nano anchor-free detector for N classes.

    Parameters
    ----------
    nc      : int   – number of object classes
    reg_max : int   – DFL bins (default 16, controls fine localisation)

    Input
    -----
    x : (B, 3, H, W)   H and W must be multiples of 32; typically 640×640.

    Output (training)
    -----------------
    dict  produced by Detect.forward() — used directly by YOLO11Loss.

    Output (inference)
    ------------------
    (predictions, raw_preds)
    predictions : (B, num_anchors, 4+nc)  — decoded boxes + scores (after sigmoid)
    """

    def __init__(self, nc: int = 80, reg_max: int = 16):
        super().__init__()
        self.nc      = nc
        self.reg_max = reg_max

        self.backbone = YOLO11nBackbone()
        self.neck     = YOLO11nNeck()
        # neck output channels match Detect head inputs: (64, 128, 256)
        self.head     = Detect(nc=nc, reg_max=reg_max, ch=(64, 128, 256))

        self._init_weights()
        self._build_strides()

    # ── initialisation helpers ──────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Keep fixed (non-trainable) conv kernels intact, e.g. DFL integral kernel.
                if not m.weight.requires_grad:
                    continue
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _reset_head_dfl_kernel(self):
        """Ensure Detect head DFL kernel is the fixed integral basis [0..reg_max-1]."""
        if not hasattr(self.head, "dfl") or not hasattr(self.head.dfl, "conv"):
            return
        with torch.no_grad():
            w = torch.arange(self.reg_max, device=self.head.dfl.conv.weight.device, dtype=torch.float32)
            self.head.dfl.conv.weight.data.copy_(w.view(1, self.reg_max, 1, 1))
            self.head.dfl.conv.weight.requires_grad_(False)

    def _build_strides(self, img_size: int = 640):
        """Compute and register strides for the Detect head."""
        self.eval()
        with torch.no_grad():
            dummy = torch.zeros(1, 3, img_size, img_size)
            p3, p4, p5 = self.backbone(dummy)
            feats = self.neck(p3, p4, p5)
            strides = [img_size / f.shape[-2] for f in feats]
        self.head.stride = torch.tensor(strides)
        # Force xyxy output during inference so that non_max_suppression and
        # the mAP evaluation code (which both expect xyxy) receive the correct format.
        self.head.xyxy = True
        # Disable end2end mode to use the simpler one2many decode path, which
        # produces consistent xyxy format throughout inference.
        self.head._end2end = False
        self._reset_head_dfl_kernel()
        self.head.bias_init()
        self.train()

    # ── forward ────────────────────────────────────────────────────────────

    def forward(self, x):
        p3, p4, p5 = self.backbone(x)
        out_s, out_m, out_l = self.neck(p3, p4, p5)
        return self.head([out_s, out_m, out_l])

    # ── convenience ────────────────────────────────────────────────────────

    def info(self, img_size: int = 640):
        """Print parameter count and estimated GFLOPs."""
        n_params = sum(p.numel() for p in self.parameters())
        print(f"YOLO11n  |  classes={self.nc}  |  params={n_params/1e6:.2f}M")
        try:
            from torchinfo import summary
            summary(self, (1, 3, img_size, img_size), depth=3, verbose=1)
        except ImportError:
            print("Install torchinfo for a detailed layer-by-layer summary.")

def letterbox(
    img: np.ndarray,
    new_shape: int | tuple[int, int] = 640,
    color: tuple[int, int, int] = (114, 114, 114),
    auto: bool = False,
    stride: int = 32,
) -> tuple[np.ndarray, float, tuple[int, int]]:
    """
    Resize *img* to *new_shape* with letterboxing (grey padding) so that the
    aspect ratio is preserved and both dims are multiples of *stride*.

    Parameters
    ----------
    img       : HxWxC uint8 BGR/RGB image
    new_shape : target resolution (square int or (h, w) tuple)
    color     : padding colour
    auto      : if True, use minimum-rectangle padding (keeps dims % stride == 0)
    stride    : model input stride (32 for YOLO11n)

    Returns
    -------
    img       : resized / padded image (uint8)
    ratio     : actual scale ratio applied
    (dw, dh)  : pixels of padding added to width / height
    """
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    h, w = img.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)

    new_unpad = (round(w * r), round(h * r))
    dw = new_shape[1] - new_unpad[0]
    dh = new_shape[0] - new_unpad[1]

    if auto:
        dw = dw % stride
        dh = dh % stride

    dw /= 2
    dh /= 2

    if (w, h) != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

    # Symmetric padding (match albumentations LetterBox)
    top = bottom = round(dh)
    left = right = round(dw)
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def preprocess(
    images: list[np.ndarray] | np.ndarray,
    img_size: int = 640,
    device: torch.device = torch.device("cpu"),
    fp16: bool = False,
) -> tuple[torch.Tensor, list[float], list[tuple[int, int]]]:
    """
    Prepare a list of BGR uint8 images for YOLO11n inference.

    Returns
    -------
    tensor  : (B, 3, img_size, img_size) float32/float16 in [0, 1]
    ratios  : list of float scale ratios (one per image)
    paddings: list of (dw, dh) padding tuples
    """
    if isinstance(images, np.ndarray) and images.ndim == 3:
        images = [images]

    batched, ratios, paddings = [], [], []
    for img in images:
        img_lb, r, pad = letterbox(img, new_shape=img_size)
        img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(img_rgb).permute(2, 0, 1).contiguous()
        batched.append(t)
        ratios.append(r)
        paddings.append(pad)

    tensor = torch.stack(batched).to(device)
    tensor = tensor.half() if fp16 else tensor.float()
    tensor /= 255.0
    return tensor, ratios, paddings


def non_max_suppression(
    prediction: torch.Tensor,
    conf_thres: float = 0.25,
    iou_thres: float  = 0.45,
    max_det:   int    = 300,
    nc:        int    = 80,
) -> list[torch.Tensor]:
    """
    Run per-class batched NMS on decoded YOLO11n predictions.

    Parameters
    ----------
    prediction : (B, 4+nc, N)  or  (B, N, 4+nc)
                 x1y1x2y2 + class scores (already sigmoid'd from Detect._inference).
                 The channels-first layout (B, 4+nc, N) produced by the Detect head
                 is automatically transposed.
    conf_thres : minimum object confidence to keep
    iou_thres  : IoU threshold for NMS
    max_det    : maximum detections returned per image
    nc         : number of classes

    Returns
    -------
    List of (n_kept, 6) tensors [x1, y1, x2, y2, conf, cls] per image.
    """
    # Normalise to (B, N, 4+nc) regardless of input layout
    if prediction.shape[1] == (4 + nc) and prediction.ndim == 3:
        prediction = prediction.permute(0, 2, 1)   # (B, 4+nc, N) → (B, N, 4+nc)

    bs = prediction.shape[0]
    output = [torch.zeros((0, 6), device=prediction.device) for _ in range(bs)]

    for i, x in enumerate(prediction):
        # x: (N, 4+nc)
        boxes_raw = x[:, :4]
        # Canonicalize to valid xyxy ordering to avoid passing malformed boxes
        # (x2<x1 or y2<y1) to NMS/evaluation.
        x1 = torch.minimum(boxes_raw[:, 0], boxes_raw[:, 2])
        y1 = torch.minimum(boxes_raw[:, 1], boxes_raw[:, 3])
        x2 = torch.maximum(boxes_raw[:, 0], boxes_raw[:, 2])
        y2 = torch.maximum(boxes_raw[:, 1], boxes_raw[:, 3])
        boxes  = torch.stack((x1, y1, x2, y2), dim=1)
        scores = x[:, 4:]                          # class scores (post-sigmoid in Detect._inference)

        conf, cls = scores.max(dim=1, keepdim=True)
        wh = boxes[:, 2:4] - boxes[:, 0:2]
        valid = (wh[:, 0] > 1.0) & (wh[:, 1] > 1.0)
        mask = (conf.squeeze(1) > conf_thres) & valid
        x = torch.cat([boxes[mask], conf[mask], cls[mask].float()], dim=1)

        if not x.shape[0]:
            continue

        # Per-class offset so boxes of different classes don't suppress each other
        offsets = x[:, 5:6] * 4096
        boxes_off = x[:, :4] + offsets

        keep = torchvision.ops.nms(boxes_off, x[:, 4], iou_thres)
        keep = keep[:max_det]
        output[i] = x[keep]

    return output

def postprocess(
    raw_dets:  list[torch.Tensor],
    ratios:    list[float],
    paddings:  list[tuple[int, int]],
    orig_shapes: list[tuple[int, int]],
) -> list[torch.Tensor]:
    """
    Rescale detected boxes back to original image coordinates.

    Parameters
    ----------
    raw_dets    : output of non_max_suppression()
    ratios      : letterbox scale ratios (from preprocess)
    paddings    : (dw, dh) padding values (from preprocess)
    orig_shapes : (h, w) of each original image

    Returns
    -------
    List of (n_kept, 6) tensors [x1, y1, x2, y2, conf, cls] in original pixel coords.
    """
    results = []
    for det, r, (dw, dh), (oh, ow) in zip(raw_dets, ratios, paddings, orig_shapes):
        if det.shape[0]:
            det[:, [0, 2]] -= dw
            det[:, [1, 3]] -= dh
            det[:, :4] /= r
            det[:, [0, 2]] = det[:, [0, 2]].clamp(0, ow)
            det[:, [1, 3]] = det[:, [1, 3]].clamp(0, oh)
        results.append(det)
    return results



def load_checkpoint(path: str, device: str | None = None) -> tuple[YOLO11n, list[str]]:
    """Load a saved YOLO11n checkpoint and return the model in eval mode."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt  = torch.load(path, map_location=device)
    model = YOLO11n(nc=ckpt["nc"]).to(device)
    model.load_state_dict(ckpt["model"])
    # Backward compatibility for older checkpoints where the fixed DFL kernel
    # may have been overwritten by generic Conv2d weight initialisation.
    model._reset_head_dfl_kernel()

    class_names = ckpt.get("class_names")
    if class_names is None:
        class_names = [str(i) for i in range(model.nc)]
    
    model.eval()
    print(f"Loaded YOLO11n (nc={ckpt['nc']}) from '{path}' (epoch {ckpt['epoch']})")
    return model, class_names
