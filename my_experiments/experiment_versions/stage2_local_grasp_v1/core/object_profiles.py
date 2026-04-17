"""Object and controller templates for stage-2 local grasp training."""

from copy import deepcopy


OBJECT_PROFILES = {
    "cube_small": {
        "object_class": "cube",
        "shape_tag": "box",
        "size_tag": "small",
        "object_family": "box_local_grasp_v1",
        "condition_feature_names": ["shape_box", "size_small", "material_default"],
        "condition_features": [1.0, 0.0, 0.0],
        "grasp_offset": [0.0, 0.0, 0.028],
        "object_half_extents": [0.02095668, 0.02134781, 0.02039102],
        "success_height": 0.05,
        "pregrasp_sampling": {
            "radius_range": [0.07, 0.1],
            "elevation_deg_range": [25.0, 60.0],
            "azimuth_center_deg": 180.0,
            "azimuth_span_deg": 90.0,
            "reference_height_offset": 0.01,
        },
        "stage_targets": {
            "stage_a": {
                "workflow_mode": "topdown_descend",
                # Stage A should terminate near the exact handoff region that
                # Stage B expects: object centered between the fingers, open
                # gripper, no table contact.
                "local_target": [0.0, 0.0, 0.0],
                "reset_strategy": "teleop_like",
                "teleop_handoff_reset": {
                    "bank_size": 6,
                    "reuse_limit": 256,
                    "cube_refresh_threshold": 0.03,
                    "bank_build_attempt_limit": 18,
                    "refine_steps": 18,
                    "handoff_xy_threshold": 0.022,
                    "handoff_vertical_threshold": 0.024,
                    "view_local_target": {
                        "x": 0.0,
                        "y": 0.0,
                        "z": 0.065,
                    },
                    "view_local_tolerance": {
                        "x": 0.010,
                        "y": 0.010,
                        "z": 0.012,
                    },
                    "joint_noise_std": [0.006, 0.010, 0.006, 0.012, 0.006, 0.012, 0.006],
                },
                "success_thresholds": {
                    "x": 0.012,
                    "y": 0.010,
                    "z": 0.014,
                },
                "cage_geometry": {
                    "fore_aft_target_x": 0.0,
                    "fore_aft_tolerance": 0.010,
                    "lateral_margin": 0.004,
                },
                "table_clearance_threshold": 0.018,
                "open_width_threshold": 0.03,
                "b_handoff_stable_steps": 3,
                "descent_control": {
                    "x_unlock": 0.012,
                    "y_unlock": 0.010,
                    "prealign_downward_limit": 0.002,
                    "forced_descend_action": -0.22,
                },
                "visual_reference": {
                    "enabled": True,
                    "progress_scale": 1.2,
                    "error_scale": 0.55,
                    "match_bonus": 0.20,
                    "match_threshold": 0.070,
                    "reference_local_target": {"x": 0.0, "y": 0.0, "z": 0.012},
                    "max_contact_retries": 4,
                },
                "workspace_bounds": {
                    "x": 0.10,
                    "y": 0.06,
                    "z_min": -0.035,
                    "z_max": 0.12,
                },
            },
            "stage_b": {
                "local_target": [0.0, 0.0, 0.0],
                "success_thresholds": {
                    "x": 0.020,
                    "y": 0.010,
                    "z": 0.014,
                },
                "cage_geometry": {
                    "fore_aft_target_x": 0.0,
                    "fore_aft_tolerance": 0.010,
                    "lateral_margin": 0.002,
                },
                "workspace_bounds": {
                    "x": 0.070,
                    "y": 0.040,
                    "z_min": -0.028,
                    "z_max": 0.028,
                },
                "reset_alignment_steps": 24,
                "reset_retry_limit": 4,
                "grasp_success_steps": 3,
            },
            "stage_c": {
                "local_target": [0.0, 0.0, 0.0],
                "success_height": 0.02,
                "post_grasp_xy_action_limit": 0.10,
                "post_grasp_rot_action_limit": 0.05,
                "upright_success_cos": 0.96,
                "severe_tilt_cos": 0.85,
                "workspace_bounds": {
                    "x": 0.080,
                    "y": 0.050,
                    "z_min": -0.030,
                    "z_max": 0.060,
                },
                "reset_close_steps": 24,
                "post_grasp_settle_steps": 4,
                "reset_retry_limit": 4,
                "lift_success_steps": 3,
                "hold_close_action": 1.0,
                "hold_bonus": 0.45,
                "lift_amount_scale": 22.0,
                "lift_progress_scale": 28.0,
                "upright_bonus_scale": 0.8,
                "tilt_penalty_scale": 1.4,
                "prelift_height": 0.012,
                "no_lift_penalty": 0.35,
                "downward_after_grasp_penalty": 0.45,
                "stall_after_grasp_penalty": 0.30,
            },
        },
        "impedance_template": {
            "template_name": "box_small_default_impedance_v1",
            "impedance_mode": "fixed",
            "kp": [180.0, 180.0, 180.0, 220.0, 220.0, 220.0],
            "damping_ratio": 1.3,
            "output_max": [0.03, 0.03, 0.03, 0.10, 0.10, 0.10],
            "output_min": [-0.03, -0.03, -0.03, -0.10, -0.10, -0.10],
        },
    }
}

OBJECT_PROFILES["cube_small_loose"] = deepcopy(OBJECT_PROFILES["cube_small"])
OBJECT_PROFILES["cube_small_loose"].update(
    {
        "size_tag": "small_loose",
        "object_half_extents": [0.0165, 0.0165, 0.0160],
        "grasp_offset": [0.0, 0.0, 0.023],
    }
)
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["local_target"][0] = 0.0
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["teleop_handoff_reset"]["view_local_target"]["x"] = 0.0
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["teleop_handoff_reset"]["view_local_target"]["z"] = 0.070
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["teleop_handoff_reset"]["view_local_tolerance"]["x"] = 0.010
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["teleop_handoff_reset"]["view_local_tolerance"]["y"] = 0.010
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["teleop_handoff_reset"]["view_local_tolerance"]["z"] = 0.012
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["visual_reference"]["reference_local_target"]["x"] = 0.0
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["cage_geometry"]["fore_aft_target_x"] = 0.0
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["cage_geometry"]["fore_aft_tolerance"] = 0.012
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["success_thresholds"]["x"] = 0.012
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["descent_control"]["x_unlock"] = 0.012
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["descent_control"]["y_unlock"] = 0.010
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["descent_control"]["prealign_downward_limit"] = 0.002
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_a"]["descent_control"]["forced_descend_action"] = -0.22
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_b"]["success_thresholds"]["x"] = 0.018
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_b"]["cage_geometry"]["fore_aft_tolerance"] = 0.012
OBJECT_PROFILES["cube_small_loose"]["stage_targets"]["stage_c"]["prelift_height"] = 0.010

LABEL_TO_PROFILE = {
    ("cube", "box", "small"): "cube_small",
    ("cube", "box", "small_loose"): "cube_small_loose",
}


def get_object_profile(name: str):
    """Returns a deep copy so callers can mutate safely."""
    if name not in OBJECT_PROFILES:
        raise KeyError(f"Unknown object profile: {name}")
    return deepcopy(OBJECT_PROFILES[name])


def resolve_object_profile_name(
    object_class: str | None = None,
    shape_tag: str | None = None,
    size_tag: str | None = None,
    fallback: str = "cube_small",
):
    """Maps vision labels to a known local-grasp object profile."""
    key = (
        object_class.lower() if isinstance(object_class, str) else None,
        shape_tag.lower() if isinstance(shape_tag, str) else None,
        size_tag.lower() if isinstance(size_tag, str) else None,
    )
    return LABEL_TO_PROFILE.get(key, fallback)
