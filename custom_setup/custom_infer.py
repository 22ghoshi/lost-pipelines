#!/usr/bin/env python3

import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from custom_arch import (
    YOLO11n,
    letterbox,
    non_max_suppression,
    postprocess
)

_PALETTE = [
    (255,  56,  56), (255, 157, 151), (255, 112,  31), (255, 178,  29),
    (207, 210,  49), ( 72, 249,  10), (146, 204,  23), ( 61, 219, 134),
    ( 26, 147,  52), (  0, 212, 187), ( 44, 153, 168), (  0, 194, 255),
    ( 52,  69, 147), (100, 115, 255), (  0,  24, 236), (132,  56, 255),
    ( 82,   0, 133), (203,  56, 255), (255, 149, 200), (255,  55, 199),
]


def _color(cls_idx: int) -> tuple[int, int, int]:
    return _PALETTE[int(cls_idx) % len(_PALETTE)]



def draw_detections(
    img_bgr:     np.ndarray,
    dets:        torch.Tensor,      # (n, 6)  x1 y1 x2 y2 conf cls
    class_names: list[str],
    thickness:   int = 2,
) -> np.ndarray:
    """Draw bounding boxes and labels on *img_bgr* (copy). Returns the annotated image."""
    out = img_bgr.copy()
    if dets is None or dets.shape[0] == 0:
        return out

    for *xyxy, conf, cls in dets.cpu().tolist():
        x1, y1, x2, y2 = map(int, xyxy)
        c     = int(cls)
        color = _color(c)
        name  = class_names[c] if c < len(class_names) else str(c)
        label = f"{name}  {conf:.2f}"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(out, (x1, y1 - th - 4), (x1 + tw, y1), color, -1)
        cv2.putText(out, label, (x1, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return out



@torch.no_grad()
def infer_image(
    model:      YOLO11n,
    img_bgr:    np.ndarray,
    device:     torch.device,
    conf_thres: float = 0.25,
    iou_thres:  float = 0.45,
    img_size:   int   = 640,
) -> torch.Tensor:
    """
    Run the full detection pipeline on one BGR uint8 image.

    Returns
    -------
    dets : (n, 6)  [x1, y1, x2, y2, conf, cls]  in original-image pixel coords.
           Empty (0, 6) tensor when no detections pass the threshold.
    """
    nc = model.nc
    orig_h, orig_w = img_bgr.shape[:2]

    # Ensure model inference uses consistent xyxy format (disable end2end mode).
    model.head._end2end = False

    # Match training/metric_eval preprocessing: letterbox to preserve aspect ratio.
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    resized_rgb, ratio, padding = letterbox(rgb, new_shape=img_size)
    tensor = (
        torch.from_numpy(resized_rgb)
        .permute(2, 0, 1)
        .contiguous()
        .unsqueeze(0)
        .to(device)
        .float()
        / 255.0
    )
    raw_out, _ = model(tensor)                # (B, N, 4+nc)  xyxy, resized-image pixel space

    nms_dets = non_max_suppression(
        raw_out, conf_thres=conf_thres, iou_thres=iou_thres, nc=nc
    )
    post_dets = postprocess(
        nms_dets,
        ratios=[ratio],
        paddings=[padding],
        orig_shapes=[(orig_h, orig_w)],
    )
    return post_dets[0].clone()              # (n, 6)
