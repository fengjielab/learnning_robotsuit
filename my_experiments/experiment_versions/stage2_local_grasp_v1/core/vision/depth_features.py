"""Feature extraction helpers for wrist RGB-D observations."""

from __future__ import annotations

import numpy as np


def _center_crop(arr: np.ndarray, crop_fraction: float) -> np.ndarray:
    if crop_fraction >= 0.999:
        return arr
    height, width = arr.shape[:2]
    crop_h = max(1, int(round(height * crop_fraction)))
    crop_w = max(1, int(round(width * crop_fraction)))
    top = max(0, (height - crop_h) // 2)
    left = max(0, (width - crop_w) // 2)
    return arr[top : top + crop_h, left : left + crop_w]


def _resize_nearest(arr: np.ndarray, out_shape: tuple[int, int]) -> np.ndarray:
    out_h, out_w = out_shape
    in_h, in_w = arr.shape[:2]
    row_idx = np.linspace(0, in_h - 1, out_h).astype(np.int32)
    col_idx = np.linspace(0, in_w - 1, out_w).astype(np.int32)
    if arr.ndim == 2:
        return arr[row_idx][:, col_idx]
    return arr[row_idx][:, col_idx, :]


def encode_depth_map(
    depth_m: np.ndarray | None,
    min_distance_m: float,
    max_distance_m: float,
    out_shape: tuple[int, int] = (16, 16),
    crop_fraction: float = 0.75,
) -> np.ndarray:
    """Encodes a metric depth map into a compact inverse-depth feature vector."""
    feature_dim = out_shape[0] * out_shape[1]
    if depth_m is None:
        return np.zeros(feature_dim, dtype=np.float32)

    depth = np.asarray(depth_m, dtype=np.float32)
    depth = np.nan_to_num(depth, nan=max_distance_m, posinf=max_distance_m, neginf=min_distance_m)
    depth = np.clip(depth, min_distance_m, max_distance_m)
    depth = _center_crop(depth, crop_fraction)
    depth = _resize_nearest(depth, out_shape)

    near_inv = 1.0 / max(min_distance_m, 1.0e-6)
    far_inv = 1.0 / max(max_distance_m, min_distance_m + 1.0e-6)
    depth_inv = 1.0 / np.maximum(depth, 1.0e-6)
    denom = max(near_inv - far_inv, 1.0e-6)
    normalized = np.clip((depth_inv - far_inv) / denom, 0.0, 1.0)
    return normalized.astype(np.float32).reshape(-1)


def encode_rgb_image(
    rgb: np.ndarray | None,
    out_shape: tuple[int, int] = (8, 8),
    crop_fraction: float = 0.75,
) -> np.ndarray:
    """Encodes an RGB image into a compact, normalized flat vector."""
    feature_dim = out_shape[0] * out_shape[1] * 3
    if rgb is None:
        return np.zeros(feature_dim, dtype=np.float32)

    rgb_arr = np.asarray(rgb, dtype=np.float32) / 255.0
    rgb_arr = _center_crop(rgb_arr, crop_fraction)
    rgb_arr = _resize_nearest(rgb_arr, out_shape)
    return rgb_arr.astype(np.float32).reshape(-1)


def encode_wrist_rgbd(
    rgb: np.ndarray | None,
    depth_m: np.ndarray | None,
    min_distance_m: float,
    max_distance_m: float,
    depth_shape: tuple[int, int] = (16, 16),
    rgb_shape: tuple[int, int] = (8, 8),
    crop_fraction: float = 0.75,
) -> tuple[np.ndarray, dict[str, int]]:
    """Builds a single flat RGB-D feature vector and returns its dimensional metadata."""
    depth_vec = encode_depth_map(
        depth_m=depth_m,
        min_distance_m=min_distance_m,
        max_distance_m=max_distance_m,
        out_shape=depth_shape,
        crop_fraction=crop_fraction,
    )
    rgb_vec = encode_rgb_image(rgb=rgb, out_shape=rgb_shape, crop_fraction=crop_fraction)
    visual_vec = np.concatenate([depth_vec, rgb_vec]).astype(np.float32)
    return visual_vec, {
        "depth_dim": int(depth_vec.size),
        "rgb_dim": int(rgb_vec.size),
        "visual_dim": int(visual_vec.size),
    }

