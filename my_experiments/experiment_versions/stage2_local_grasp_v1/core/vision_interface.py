"""Vision label interface for stage-2 local grasp experiments."""

from __future__ import annotations

import json
import os
from copy import deepcopy

from core.camera_profiles import get_camera_profile
from core.object_profiles import get_object_profile, resolve_object_profile_name

REQUIRED_VISION_KEYS = (
    "object_class",
    "shape_tag",
    "size_tag",
    "detection_confidence",
    "source",
)


def _normalize_label(value):
    if value is None:
        return None
    return str(value).strip().lower()


def _default_labels_from_profile(object_profile_name: str):
    profile = get_object_profile(object_profile_name)
    return {
        "object_class": profile["object_class"],
        "shape_tag": profile["shape_tag"],
        "size_tag": profile["size_tag"],
        "object_profile_name": object_profile_name,
        "detection_confidence": 1.0,
        "source": "manual_default",
    }


def _validate_detection_confidence(value):
    try:
        confidence = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"detection_confidence 必须是数值，当前收到: {value!r}") from exc

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(f"detection_confidence 必须位于 [0, 1]，当前收到: {confidence}")
    return confidence


def _build_grasp_condition(labels: dict, fallback_object_profile: str):
    normalized_labels = {
        "object_class": _normalize_label(labels.get("object_class")),
        "shape_tag": _normalize_label(labels.get("shape_tag")),
        "size_tag": _normalize_label(labels.get("size_tag")),
        "object_profile_name": labels.get("object_profile_name"),
        "detection_confidence": _validate_detection_confidence(labels.get("detection_confidence", 1.0)),
        "source": str(labels.get("source", "vision_input_json")).strip() or "vision_input_json",
    }

    if normalized_labels["object_profile_name"] is None:
        normalized_labels["object_profile_name"] = resolve_object_profile_name(
            object_class=normalized_labels["object_class"],
            shape_tag=normalized_labels["shape_tag"],
            size_tag=normalized_labels["size_tag"],
            fallback=fallback_object_profile,
        )

    if normalized_labels["shape_tag"] != "box":
        raise ValueError(
            "当前分层抓取 v1 只支持 shape_tag='box'。"
            f" 收到 shape_tag={normalized_labels['shape_tag']!r}"
        )

    profile_name = normalized_labels["object_profile_name"]
    profile = get_object_profile(profile_name)
    if _normalize_label(profile.get("shape_tag")) != normalized_labels["shape_tag"]:
        raise ValueError(
            "vision_input 与 object_profile 不一致："
            f" profile={profile_name}, profile.shape_tag={profile.get('shape_tag')!r},"
            f" input.shape_tag={normalized_labels['shape_tag']!r}"
        )
    if _normalize_label(profile.get("size_tag")) != normalized_labels["size_tag"]:
        raise ValueError(
            "vision_input 与 object_profile 不一致："
            f" profile={profile_name}, profile.size_tag={profile.get('size_tag')!r},"
            f" input.size_tag={normalized_labels['size_tag']!r}"
        )

    grasp_condition = {
        "object_profile_name": profile_name,
        "object_class": profile["object_class"],
        "shape_tag": profile["shape_tag"],
        "size_tag": profile["size_tag"],
        "condition_features": deepcopy(profile["condition_features"]),
        "impedance_template": deepcopy(profile["impedance_template"]),
        "stage_targets": deepcopy(profile["stage_targets"]),
        "supported_object_family": "box_local_grasp_v1",
        "detection_confidence": normalized_labels["detection_confidence"],
        "source": normalized_labels["source"],
    }
    return normalized_labels, grasp_condition


def load_vision_input(vision_input_path: str | None):
    """Loads a vision label JSON file if present."""
    if not vision_input_path:
        return None

    resolved_path = os.path.abspath(vision_input_path)
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    payload["vision_input_path"] = resolved_path
    return payload


def resolve_stage2_context(
    camera_profile_name: str = "realsense_d435i",
    vision_input_path: str | None = None,
    fallback_object_profile: str = "cube_small",
):
    """
    Resolves the camera profile and vision labels into a single training / reset context.

    Expected JSON keys for @vision_input_path:
    - object_class
    - shape_tag
    - size_tag
    - object_profile_name (optional override)
    - detection_confidence (optional)
    - source (optional)
    """
    camera_profile = get_camera_profile(camera_profile_name)
    vision_payload = load_vision_input(vision_input_path)

    if vision_payload is None:
        labels = _default_labels_from_profile(fallback_object_profile)
        labels, grasp_condition = _build_grasp_condition(labels, fallback_object_profile)
    else:
        missing = [key for key in REQUIRED_VISION_KEYS if key not in vision_payload]
        if missing:
            raise ValueError(
                "vision_input JSON 缺少必填字段："
                f" {missing}。当前 v1 需要固定字段 {list(REQUIRED_VISION_KEYS)}"
            )
        labels, grasp_condition = _build_grasp_condition(
            {
                "object_class": vision_payload.get("object_class"),
                "shape_tag": vision_payload.get("shape_tag"),
                "size_tag": vision_payload.get("size_tag"),
                "object_profile_name": vision_payload.get("object_profile_name"),
                "detection_confidence": vision_payload.get("detection_confidence"),
                "source": vision_payload.get("source"),
            },
            fallback_object_profile=fallback_object_profile,
        )

    return {
        "camera_profile_name": camera_profile_name,
        "camera_profile": camera_profile,
        "vision_labels": deepcopy(labels),
        "vision_input_path": os.path.abspath(vision_input_path) if vision_input_path else None,
        "object_profile_name": labels["object_profile_name"],
        "grasp_condition": deepcopy(grasp_condition),
        "policy_uses_visual": False,
        "vision_role": "classification_only",
    }
