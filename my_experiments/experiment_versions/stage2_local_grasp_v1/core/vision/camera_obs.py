"""Utilities for extracting wrist camera observations from robosuite."""

from __future__ import annotations

from typing import Any

import numpy as np
from robosuite.utils.camera_utils import get_real_depth_map


def camera_obs_keys(camera_name: str) -> tuple[str, str]:
    """Returns the RGB and depth observation keys for a named camera."""
    return f"{camera_name}_image", f"{camera_name}_depth"


def extract_camera_observation(obs: dict[str, Any], camera_name: str) -> dict[str, Any]:
    """Pulls RGB and depth arrays for a named camera out of a robosuite observation dict."""
    rgb_key, depth_key = camera_obs_keys(camera_name)
    rgb = obs.get(rgb_key)
    depth = obs.get(depth_key)
    return {
        "camera_name": camera_name,
        "rgb_key": rgb_key,
        "depth_key": depth_key,
        "rgb": None if rgb is None else np.asarray(rgb, dtype=np.uint8),
        "depth": None if depth is None else np.asarray(depth, dtype=np.float32),
    }


def depth_map_to_meters(sim, depth_map: np.ndarray | None) -> np.ndarray | None:
    """Converts MuJoCo normalized depth to metric depth in meters."""
    if depth_map is None:
        return None
    depth = np.asarray(depth_map, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    # Some MuJoCo backends return tiny numeric spillover outside [0, 1].
    # Clamp defensively before converting to metric depth.
    depth = np.nan_to_num(depth, nan=1.0, posinf=1.0, neginf=0.0)
    depth = np.clip(depth, 0.0, 1.0)
    return get_real_depth_map(sim=sim, depth_map=depth).astype(np.float32)


def summarize_camera_observation(camera_obs: dict[str, Any]) -> dict[str, Any]:
    """Returns lightweight metadata for logging/debugging without copying full image arrays."""
    rgb = camera_obs.get("rgb")
    depth = camera_obs.get("depth")
    depth_m = camera_obs.get("depth_m")
    summary = {
        "camera_name": camera_obs.get("camera_name"),
        "has_rgb": rgb is not None,
        "has_depth": depth is not None,
        "rgb_shape": tuple(rgb.shape) if rgb is not None else None,
        "depth_shape": tuple(depth.shape) if depth is not None else None,
    }
    if depth_m is not None and depth_m.size:
        summary.update(
            {
                "depth_min_m": float(np.nanmin(depth_m)),
                "depth_max_m": float(np.nanmax(depth_m)),
                "depth_mean_m": float(np.nanmean(depth_m)),
            }
        )
    return summary
