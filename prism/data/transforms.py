"""Separate student/teacher preprocessing and coordinate-preserving RoI sampling."""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import Tensor


def center_crop_box(height: int, width: int) -> tuple[int, int, int, int]:
    """Integer centered square in source coordinates (left, top, right, bottom)."""
    side = min(height, width)
    if side <= 0:
        raise ValueError("Expected positive image dimensions")
    left, top = (width - side) // 2, (height - side) // 2
    return left, top, left + side, top + side


def resize_inputs(images: Tensor, teacher_size: int = 518, *,
                  center_crop: bool = False) -> tuple[Tensor, Tensor]:
    from torch.nn import functional as F

    if teacher_size <= 0 or teacher_size % 14:
        raise ValueError("Teacher input size must be a positive multiple of 14")
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("Expected RGB [B,3,H,W]")
    teacher = images
    if center_crop:
        left, top, right, bottom = center_crop_box(*images.shape[-2:])
        teacher = images[..., top:bottom, left:right]
    return (F.interpolate(images, (256, 256), mode="bilinear", align_corners=False, antialias=True),
            F.interpolate(teacher, (teacher_size, teacher_size), mode="bilinear", align_corners=False,
                          antialias=True))


def sample_roi_targets(features: Tensor, rois: Tensor, *, image_height: int,
                       image_width: int, output_size: int = 1) -> Tensor:
    from torchvision.ops import roi_align
    if rois.ndim != 2 or rois.shape[1] != 5:
        raise ValueError("Expected RoIs [N,5] as batch_index,x1,y1,x2,y2 in source pixels")
    scaled = rois.clone().to(features)
    scaled[:, [1, 3]] *= features.shape[-1] / image_width
    scaled[:, [2, 4]] *= features.shape[-2] / image_height
    return roi_align(features, scaled, output_size=output_size, spatial_scale=1.0,
                     sampling_ratio=2, aligned=True)
