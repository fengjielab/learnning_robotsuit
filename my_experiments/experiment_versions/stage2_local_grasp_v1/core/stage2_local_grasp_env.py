"""Stage-2 local grasp environment built on top of robosuite Lift."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Dict

import gymnasium as gym
import numpy as np
import robosuite as suite
import robosuite.utils.transform_utils as T
from gymnasium import spaces

from core.camera_profiles import get_camera_profile
from core.object_profiles import get_object_profile
from core.vision.camera_obs import (
    depth_map_to_meters,
    extract_camera_observation,
    summarize_camera_observation,
)
from core.vision.depth_features import encode_wrist_rgbd
from core.vision.handoff_gate import StageAHandoffGate


class Stage2LocalGraspEnv(gym.Env):
    """
    Local grasp stage for Panda + Lift.

    Reset policy:
    1. Place cube at a fixed or slightly randomized tabletop pose.
    2. Sample a pregrasp position from an upper hemisphere around the cube.
    3. Use an IK-controlled helper env to move the end effector near that pregrasp.
    4. Copy the resulting arm joint positions into the main OSC env and start RL there.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        object_profile_name: str = "cube_small",
        object_policy: str = "fixed",
        vision_context: dict | None = None,
        has_renderer: bool = False,
        control_freq: int = 20,
        horizon: int = 220,
        seed: int | None = None,
        ik_retry_limit: int = 10,
        reset_ik_steps: int = 80,
    ):
        super().__init__()

        if object_policy not in {"fixed", "small_random"}:
            raise ValueError(f"Unsupported object_policy: {object_policy}")

        self.vision_context = deepcopy(vision_context) if vision_context else None
        if self.vision_context and self.vision_context.get("object_profile_name"):
            object_profile_name = self.vision_context["object_profile_name"]

        self.object_profile_name = object_profile_name
        self.object_profile = get_object_profile(object_profile_name)
        self.object_policy = object_policy
        self.has_renderer = has_renderer
        self.control_freq = control_freq
        self.horizon = horizon
        self.ik_retry_limit = ik_retry_limit
        self.reset_ik_steps = reset_ik_steps

        self.rng = np.random.default_rng(seed)
        self.condition_features = np.array(
            self.object_profile["condition_features"], dtype=np.float32
        )
        self.grasp_condition = {
            "object_profile_name": self.object_profile_name,
            "object_class": self.object_profile["object_class"],
            "shape_tag": self.object_profile["shape_tag"],
            "size_tag": self.object_profile["size_tag"],
            "condition_features": deepcopy(self.object_profile["condition_features"]),
            "impedance_template": deepcopy(self.object_profile["impedance_template"]),
            "stage_targets": deepcopy(self.object_profile["stage_targets"]),
            "supported_object_family": self.object_profile.get("object_family", "box_local_grasp_v1"),
            "detection_confidence": 1.0,
            "source": "manual_default",
        }

        if self.vision_context is None:
            default_camera_profile_name = "realsense_d435i"
            default_camera_profile = get_camera_profile(default_camera_profile_name)
            self.vision_context = {
                "camera_profile_name": default_camera_profile_name,
                "camera_profile": default_camera_profile,
                "vision_labels": {
                    "object_class": self.object_profile["object_class"],
                    "shape_tag": self.object_profile["shape_tag"],
                    "size_tag": self.object_profile["size_tag"],
                    "object_profile_name": self.object_profile_name,
                    "detection_confidence": 1.0,
                    "source": "manual_default",
                },
                "vision_input_path": None,
                "object_profile_name": self.object_profile_name,
            }
        if self.vision_context.get("grasp_condition"):
            self.grasp_condition = deepcopy(self.vision_context["grasp_condition"])
        self.vision_context["grasp_condition"] = deepcopy(self.grasp_condition)
        self.camera_profile_name = self.vision_context.get("camera_profile_name") or "realsense_d435i"
        raw_camera_profile = self.vision_context.get("camera_profile")
        self.camera_profile = deepcopy(raw_camera_profile) if raw_camera_profile else get_camera_profile(self.camera_profile_name)
        self.vision_context["camera_profile_name"] = self.camera_profile_name
        self.vision_context["camera_profile"] = deepcopy(self.camera_profile)
        self.policy_uses_visual = bool(self.vision_context.get("policy_uses_visual", False))
        self.vision_role = str(self.vision_context.get("vision_role", "classification_only"))
        self.camera_name = "robot0_eye_in_hand"
        self.depth_stream_config = deepcopy(self.camera_profile["streams"]["depth"])
        self.color_stream_config = deepcopy(self.camera_profile["streams"]["color"])
        self.camera_height = int(self.depth_stream_config["height"])
        self.camera_width = int(self.depth_stream_config["width"])
        self.depth_min_distance_m = float(self.depth_stream_config["min_distance_m"])
        self.depth_max_distance_m = float(self.depth_stream_config["max_distance_m"])
        self.visual_depth_shape = (16, 16)
        self.visual_rgb_shape = (8, 8)
        self.visual_crop_fraction = 0.75
        self.visual_feature_dim = self.visual_depth_shape[0] * self.visual_depth_shape[1] + self.visual_rgb_shape[0] * self.visual_rgb_shape[1] * 3
        self.policy_visual_feature_dim = self.visual_feature_dim if self.policy_uses_visual else 0
        self.latest_camera_obs = {}
        self.latest_visual_feature_vector = np.zeros(self.visual_feature_dim, dtype=np.float32)
        self.latest_camera_summary = {}
        self.default_controller_orientation = None

        self.osc_controller_config = self._build_osc_controller_config()
        self.ik_controller_config = suite.load_controller_config(default_controller="IK_POSE")

        self.env = self._make_env(
            controller_configs=self.osc_controller_config,
            has_renderer=has_renderer,
        )
        self._apply_object_profile_to_env(self.env)
        self._apply_camera_profile_to_env(self.env)
        self.reset_helper_mode = "ik_pose"
        try:
            self.reset_helper_env = self._make_env(
                controller_configs=self.ik_controller_config,
                has_renderer=False,
            )
            self._apply_object_profile_to_env(self.reset_helper_env)
            self._apply_camera_profile_to_env(self.reset_helper_env)
        except Exception as exc:
            if "pybullet" not in str(exc).lower():
                raise
            self.reset_helper_mode = "osc_pose_fallback"
            helper_config = self._build_reset_helper_controller_config()
            self.reset_helper_env = self._make_env(
                controller_configs=helper_config,
                has_renderer=False,
            )
            self._apply_object_profile_to_env(self.reset_helper_env)
            self._apply_camera_profile_to_env(self.reset_helper_env)

        self.default_arm_qpos = np.array(self.env.robots[0].robot_model.init_qpos, dtype=np.float64)
        self.cube_joint = self.env.cube.joints[0]
        self.reset_helper_cube_joint = self.reset_helper_env.cube.joints[0]
        self.table_height = float(self.env.model.mujoco_arena.table_offset[2])

        self.reset_count = 0
        self.reset_successes = 0
        self.reset_failures = 0
        self.last_reset_meta: Dict[str, object] = {}
        self.last_raw_obs = None
        self.current_target_pos = None
        self.current_target_quat = None
        self.current_cube_pos = None
        self.reset_cube_height = None
        self.prev_grasp_dist = None
        self.prev_grasp_xy_dist = None
        self.prev_vertical_error_abs = None
        self.prev_lift_amount = None
        self.table_guard_height = self.table_height + 0.045
        self.grasp_zone_xy_threshold = 0.018
        self.grasp_zone_z_threshold = 0.02
        self.stage_a_handoff_gate = StageAHandoffGate(
            xy_threshold=0.025,
            vertical_threshold=0.028,
            open_threshold=0.03,
            stable_frames=8,
        )
        self.last_stage_a_handoff = self.stage_a_handoff_gate.evaluate(
            grasp_xy_dist=np.inf,
            vertical_error=np.inf,
            gripper_width=0.0,
            any_table_contact=False,
            any_contact=False,
            grasped=False,
            update_streak=False,
        )

        obs_example, _ = self.reset()
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=obs_example.shape,
            dtype=np.float32,
        )
        low, high = self.env.action_spec
        self.action_space = spaces.Box(
            low=np.asarray(low, dtype=np.float32),
            high=np.asarray(high, dtype=np.float32),
            dtype=np.float32,
        )

    def _make_env(self, controller_configs, has_renderer):
        return suite.make(
            "Lift",
            robots="Panda",
            controller_configs=controller_configs,
            has_renderer=has_renderer,
            has_offscreen_renderer=True,
            use_camera_obs=self.policy_uses_visual,
            use_object_obs=True,
            reward_shaping=False,
            control_freq=self.control_freq,
            horizon=self.horizon,
            hard_reset=False,
            camera_names=[self.camera_name],
            camera_heights=[self.camera_height],
            camera_widths=[self.camera_width],
            camera_depths=[True],
        )

    def _apply_object_profile_to_env(self, env):
        object_half_extents = np.array(
            self.object_profile.get("object_half_extents", [0.02095668, 0.02134781, 0.02039102]),
            dtype=np.float64,
        )
        cube_geom_names = list(env.cube.contact_geoms) + list(env.cube.visual_geoms)
        for geom_name in cube_geom_names:
            geom_id = env.sim.model.geom_name2id(geom_name)
            env.sim.model.geom_size[geom_id] = object_half_extents
        env.sim.forward()

    def _apply_camera_profile_to_env(self, env):
        extrinsics = self.camera_profile.get("extrinsics_reference_for_sim_alignment")
        if not extrinsics:
            return

        translation = extrinsics.get("ee_to_camera_translation_m")
        quat_xyzw = extrinsics.get("ee_to_camera_quat_xyzw")
        if translation is None or quat_xyzw is None:
            return

        cam_id = env.sim.model.camera_name2id(self.camera_name)
        parent_body_id = env.sim.model.cam_bodyid[cam_id]

        validated_local_mount = extrinsics.get("validated_mujoco_local_mount")
        if validated_local_mount:
            env.sim.model.cam_pos[cam_id] = np.asarray(
                validated_local_mount["cam_pos_m"], dtype=np.float64
            )
            env.sim.model.cam_quat[cam_id] = np.asarray(
                validated_local_mount["cam_quat_wxyz"], dtype=np.float64
            )
            env.sim.forward()
            return

        quat_xyzw = np.asarray(quat_xyzw, dtype=np.float64)
        ee_to_camera_rot = T.quat2mat(quat_xyzw)
        ee_to_camera_pos = np.asarray(translation, dtype=np.float64)

        mount_override = extrinsics.get("robosuite_mount_frame_override")
        if mount_override:
            body_to_ee_pos = np.asarray(
                mount_override["body_to_reference_translation_m"], dtype=np.float64
            )
            override_quat_wxyz = np.asarray(
                mount_override["body_to_reference_quat_wxyz"], dtype=np.float64
            )
            override_quat_xyzw = np.array(
                [override_quat_wxyz[1], override_quat_wxyz[2], override_quat_wxyz[3], override_quat_wxyz[0]],
                dtype=np.float64,
            )
            body_to_ee_rot = T.quat2mat(override_quat_xyzw)
        else:
            body_to_ee_pos = None
            body_to_ee_rot = None

        if body_to_ee_pos is None or body_to_ee_rot is None:
            reference_site_key = extrinsics.get("reference_site_key")
            reference_orientation_body = extrinsics.get("reference_orientation_body")
            if reference_site_key:
                site_name = env.robots[0].gripper.important_sites.get(reference_site_key)
            else:
                site_name = None

            body_world_rot = env.sim.data.body_xmat[parent_body_id].reshape(3, 3).copy()
            body_world_pos = env.sim.data.body_xpos[parent_body_id].copy()

            if site_name is not None:
                site_id = env.sim.model.site_name2id(site_name)
                site_world_rot = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
                site_world_pos = env.sim.data.site_xpos[site_id].copy()
                body_to_ee_pos = body_world_rot.T @ (site_world_pos - body_world_pos)
                if reference_orientation_body == "gripper_root_body":
                    orient_body_name = env.robots[0].gripper.root_body
                    orient_body_id = env.sim.model.body_name2id(orient_body_name)
                    orient_world_rot = env.sim.data.body_xmat[orient_body_id].reshape(3, 3).copy()
                    body_to_ee_rot = body_world_rot.T @ orient_world_rot
                else:
                    body_to_ee_rot = body_world_rot.T @ site_world_rot
            else:
                body_to_ee_rot = np.eye(3, dtype=np.float64)
                body_to_ee_pos = np.zeros(3, dtype=np.float64)

        frame_conversion = extrinsics.get("mujoco_camera_frame_conversion")
        if frame_conversion == "optical_to_mujoco_rx_pi":
            optical_to_mujoco_rot = np.diag([1.0, -1.0, -1.0])
        else:
            optical_to_mujoco_rot = np.eye(3, dtype=np.float64)

        body_to_camera_pos = body_to_ee_pos + body_to_ee_rot @ ee_to_camera_pos
        body_to_camera_rot = body_to_ee_rot @ ee_to_camera_rot @ optical_to_mujoco_rot

        env.sim.model.cam_pos[cam_id] = body_to_camera_pos
        cam_quat_xyzw = T.mat2quat(body_to_camera_rot)
        env.sim.model.cam_quat[cam_id] = np.array(
            [cam_quat_xyzw[3], cam_quat_xyzw[0], cam_quat_xyzw[1], cam_quat_xyzw[2]],
            dtype=np.float64,
        )
        env.sim.forward()

    def _build_osc_controller_config(self):
        controller = suite.load_controller_config(default_controller="OSC_POSE")
        template = self.object_profile["impedance_template"]
        controller["impedance_mode"] = template["impedance_mode"]
        controller["kp"] = template["kp"]
        controller["damping_ratio"] = template["damping_ratio"]
        controller["output_max"] = template["output_max"]
        controller["output_min"] = template["output_min"]
        controller["control_delta"] = True
        return controller

    def _build_reset_helper_controller_config(self):
        controller = suite.load_controller_config(default_controller="OSC_POSE")
        controller["kp"] = [220.0, 220.0, 220.0, 160.0, 160.0, 160.0]
        controller["damping_ratio"] = 1.1
        controller["control_delta"] = True
        controller["output_max"] = [0.02, 0.02, 0.02, 0.18, 0.18, 0.18]
        controller["output_min"] = [-0.02, -0.02, -0.02, -0.18, -0.18, -0.18]
        return controller

    def _sample_cube_position(self):
        base = np.array([0.0, 0.0, self.table_height + 0.01], dtype=np.float64)
        if self.object_policy == "fixed":
            return base
        jitter = self.rng.uniform(low=[-0.015, -0.015], high=[0.015, 0.015], size=2)
        base[:2] += jitter
        return base

    def _sample_pregrasp_target(self, cube_pos):
        sampling = self.object_profile["pregrasp_sampling"]
        radius = self.rng.uniform(*sampling["radius_range"])
        elev = math.radians(self.rng.uniform(*sampling["elevation_deg_range"]))
        az_center = math.radians(sampling["azimuth_center_deg"])
        az_span = math.radians(sampling["azimuth_span_deg"] / 2.0)
        azim = self.rng.uniform(az_center - az_span, az_center + az_span)

        direction = np.array(
            [
                math.cos(elev) * math.cos(azim),
                math.cos(elev) * math.sin(azim),
                math.sin(elev),
            ],
            dtype=np.float64,
        )
        reference = cube_pos.copy()
        reference[2] += sampling["reference_height_offset"]
        return reference + radius * direction

    def _set_free_joint_pose(self, env, joint_name, pos):
        joint_qpos = np.array(env.sim.data.get_joint_qpos(joint_name), dtype=np.float64)
        joint_qpos[:3] = pos
        env.sim.data.set_joint_qpos(joint_name, joint_qpos)
        qvel_addr = env.sim.model.get_joint_qvel_addr(joint_name)
        if isinstance(qvel_addr, tuple):
            env.sim.data.qvel[qvel_addr[0] : qvel_addr[1]] = 0.0
        else:
            env.sim.data.qvel[qvel_addr] = 0.0

    def _set_gripper_fully_open(self, env):
        robot = env.robots[0]
        gripper_qpos_idx = getattr(robot, "_ref_gripper_joint_pos_indexes", None)
        if gripper_qpos_idx is None:
            return
        env.sim.data.qpos[gripper_qpos_idx] = np.array(robot.gripper.init_qpos, dtype=np.float64)
        gripper_qvel_idx = getattr(robot, "_ref_gripper_joint_vel_indexes", None)
        if gripper_qvel_idx is not None:
            env.sim.data.qvel[gripper_qvel_idx] = 0.0

    def _quat_axis_error(self, target_quat, current_quat):
        quat_error = T.quat_distance(target_quat, current_quat)
        return T.quat2axisangle(quat_error)

    def _build_zero_visual_features(self):
        return np.zeros(self.policy_visual_feature_dim, dtype=np.float32)

    def _get_camera_observation(self, obs, sim=None):
        camera_obs = extract_camera_observation(obs, self.camera_name)
        sim = self.env.sim if sim is None else sim
        if camera_obs["depth"] is not None:
            camera_obs["depth_m"] = depth_map_to_meters(sim=sim, depth_map=camera_obs["depth"])
        else:
            camera_obs["depth_m"] = None
        self.latest_camera_obs = camera_obs
        self.latest_camera_summary = summarize_camera_observation(camera_obs)
        return camera_obs

    def _build_visual_features(self, obs):
        if not self.policy_uses_visual:
            self.latest_visual_feature_vector = np.zeros(self.visual_feature_dim, dtype=np.float32)
            self.latest_camera_obs = {}
            self.latest_camera_summary = {}
            return self._build_zero_visual_features()
        camera_obs = self._get_camera_observation(obs)
        visual_vec, _ = encode_wrist_rgbd(
            rgb=camera_obs.get("rgb"),
            depth_m=camera_obs.get("depth_m"),
            min_distance_m=self.depth_min_distance_m,
            max_distance_m=self.depth_max_distance_m,
            depth_shape=self.visual_depth_shape,
            rgb_shape=self.visual_rgb_shape,
            crop_fraction=self.visual_crop_fraction,
        )
        self.latest_visual_feature_vector = visual_vec.astype(np.float32)
        return self.latest_visual_feature_vector

    def _update_progress_trackers(self, obs):
        grasp_goal, grasp_delta, grasp_xy_dist, vertical_error, _ = self._get_grasp_zone_state(obs)
        self.prev_grasp_dist = float(np.linalg.norm(grasp_delta))
        self.prev_grasp_xy_dist = float(grasp_xy_dist)
        self.prev_vertical_error_abs = float(abs(vertical_error))
        return {
            "grasp_goal": grasp_goal,
            "grasp_delta": grasp_delta,
            "grasp_xy_dist": grasp_xy_dist,
            "vertical_error": vertical_error,
        }

    def evaluate_stage_a_handoff(self, obs=None, update_streak=False):
        obs = self.last_raw_obs if obs is None else obs
        if obs is None:
            return self.last_stage_a_handoff

        stage = self._get_stage_state(obs)
        handoff = self.stage_a_handoff_gate.evaluate(
            grasp_xy_dist=stage["grasp_xy_dist"],
            vertical_error=stage["vertical_error"],
            gripper_width=stage["gripper_width"],
            any_table_contact=stage["any_table_contact"],
            any_contact=stage["any_contact"],
            grasped=stage["grasped"],
            update_streak=update_streak,
        )
        self.last_stage_a_handoff = handoff
        return handoff

    def sync_from_raw_observation(self, obs, update_handoff_streak=False):
        self.last_raw_obs = obs
        self.evaluate_stage_a_handoff(obs, update_streak=update_handoff_streak)
        return self._build_observation(obs)

    def prime_policy_from_raw_observation(self, obs, update_handoff_streak=False):
        self.last_raw_obs = obs
        self.prev_lift_amount = 0.0 if self.prev_lift_amount is None else self.prev_lift_amount
        self._update_progress_trackers(obs)
        self.evaluate_stage_a_handoff(obs, update_streak=update_handoff_streak)
        return self._build_observation(obs)

    def _solve_ik_pregrasp(self, cube_pos):
        self.reset_helper_env.robots[0].init_qpos = self.default_arm_qpos.copy()
        obs = self.reset_helper_env.reset()
        self._set_gripper_fully_open(self.reset_helper_env)
        self._set_free_joint_pose(self.reset_helper_env, self.reset_helper_cube_joint, cube_pos)
        self.reset_helper_env.sim.forward()
        obs = self.reset_helper_env._get_observations(force_update=True)

        if self.default_controller_orientation is None:
            self.default_controller_orientation = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        target_quat = self.default_controller_orientation.copy()

        for attempt in range(1, self.ik_retry_limit + 1):
            target_pos = self._sample_pregrasp_target(cube_pos)
            rollout_obs = obs
            success = False

            for _ in range(self.reset_ik_steps):
                eef_pos = np.array(rollout_obs["robot0_eef_pos"], dtype=np.float64)
                eef_quat = np.array(rollout_obs["robot0_eef_quat"], dtype=np.float64)
                pos_error = target_pos - eef_pos
                ori_error = self._quat_axis_error(target_quat, eef_quat)

                action = np.zeros(self.reset_helper_env.action_dim, dtype=np.float32)
                action[:3] = np.clip(pos_error / 0.02, -1.0, 1.0)
                action[3:6] = np.clip(ori_error / 0.10, -1.0, 1.0)
                action[-1] = -1.0

                rollout_obs, _, _, _ = self.reset_helper_env.step(action)
                final_pos_error = np.linalg.norm(target_pos - rollout_obs["robot0_eef_pos"])
                final_ori_error = np.linalg.norm(
                    self._quat_axis_error(
                        target_quat,
                        np.array(rollout_obs["robot0_eef_quat"], dtype=np.float64),
                    )
                )
                if final_pos_error < 0.012 and final_ori_error < 0.20:
                    success = True
                    break

            if success:
                arm_qpos = np.array(
                    self.reset_helper_env.sim.data.qpos[self.reset_helper_env.robots[0]._ref_joint_pos_indexes],
                    dtype=np.float64,
                )
                achieved_pos = np.array(rollout_obs["robot0_eef_pos"], dtype=np.float64)
                achieved_quat = np.array(rollout_obs["robot0_eef_quat"], dtype=np.float64)
                return {
                    "success": True,
                    "attempt": attempt,
                    "arm_qpos": arm_qpos,
                    "target_pos": achieved_pos,
                    "target_quat": achieved_quat,
                }

        return {
            "success": False,
            "attempt": self.ik_retry_limit,
            "arm_qpos": self.default_arm_qpos.copy(),
            "target_pos": None,
            "target_quat": self.default_controller_orientation.copy(),
        }

    def _finalize_reset(
        self,
        *,
        cube_pos,
        arm_qpos,
        target_pos,
        target_quat,
        reset_success,
        reset_attempts,
        reset_strategy,
        extra_meta=None,
    ):
        self.env.robots[0].init_qpos = np.array(arm_qpos, dtype=np.float64)
        self.env.reset()
        self._set_gripper_fully_open(self.env)
        self._set_free_joint_pose(self.env, self.cube_joint, cube_pos)
        self.env.sim.forward()
        obs = self.env._get_observations(force_update=True)

        self.current_cube_pos = np.array(cube_pos, dtype=np.float64)
        if target_pos is None:
            self.current_target_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        else:
            self.current_target_pos = np.array(target_pos, dtype=np.float64)
        if target_quat is None:
            self.current_target_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        else:
            self.current_target_quat = np.array(target_quat, dtype=np.float64)
        self.reset_cube_height = float(obs["cube_pos"][2])
        self.prev_lift_amount = 0.0

        obs_vec = self.prime_policy_from_raw_observation(obs, update_handoff_streak=True)
        handoff_info = self.last_stage_a_handoff
        reset_meta = {
            "cube_pos": self.current_cube_pos.tolist(),
            "ik_success": bool(reset_success),
            "pregrasp_success": bool(reset_success),
            "ik_attempts": int(reset_attempts),
            "reset_attempts": int(reset_attempts),
            "target_pos": self.current_target_pos.tolist(),
            "reset_strategy": str(reset_strategy),
            "reset_ik_success_rate": self.reset_successes / max(self.reset_count, 1),
            "pregrasp_success_rate": self.reset_successes / max(self.reset_count, 1),
            "object_profile": self.object_profile_name,
            "object_policy": self.object_policy,
            "reset_helper_mode": self.reset_helper_mode,
            "camera_profile_name": self.vision_context.get("camera_profile_name"),
            "camera_name": self.camera_name,
            "camera_resolution": [self.camera_height, self.camera_width],
            "visual_feature_dim": self.policy_visual_feature_dim,
            "policy_uses_visual": bool(self.policy_uses_visual),
            "vision_role": self.vision_role,
            "vision_source": self.vision_context.get("vision_labels", {}).get("source"),
            "vision_detection_confidence": self.vision_context.get("vision_labels", {}).get("detection_confidence"),
            "vision_object_class": self.vision_context.get("vision_labels", {}).get("object_class"),
            "vision_shape_tag": self.vision_context.get("vision_labels", {}).get("shape_tag"),
            "vision_size_tag": self.vision_context.get("vision_labels", {}).get("size_tag"),
            "grasp_condition_family": self.grasp_condition.get("supported_object_family"),
            "impedance_template_name": self.grasp_condition.get("impedance_template", {}).get("template_name"),
            "can_handoff_to_stage_a": bool(handoff_info["can_handoff_to_stage_a"]),
            "stage_a_handoff_ready_now": bool(handoff_info["ready_now"]),
            "stage_a_handoff_consecutive_hits": int(handoff_info["consecutive_hits"]),
            "stage_a_handoff_failure_reasons": list(handoff_info["failure_reasons"]),
        }
        if extra_meta:
            reset_meta.update(extra_meta)
        self.last_reset_meta = reset_meta
        return obs_vec, dict(self.last_reset_meta)

    def _build_observation(self, obs):
        eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float32)
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float32)
        gripper_qpos = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)
        gripper_to_cube = np.array(obs["gripper_to_cube_pos"], dtype=np.float32)
        grasp_offset = np.array(self.object_profile["grasp_offset"], dtype=np.float32)
        grasp_goal = cube_pos + grasp_offset

        target_offset = grasp_goal - eef_pos
        orientation_error = self._quat_axis_error(self.current_target_quat, eef_quat).astype(np.float32)
        cube_height = np.array([cube_pos[2] - self.table_height], dtype=np.float32)
        gripper_opening = np.array([float(np.mean(np.abs(gripper_qpos)))], dtype=np.float32)
        grasped = np.array([1.0 if self.env._check_grasp(self.env.robots[0].gripper, self.env.cube) else 0.0], dtype=np.float32)

        return np.concatenate(
            [
                gripper_to_cube,
                target_offset,
                orientation_error,
                gripper_opening,
                cube_height,
                grasped,
                self.condition_features,
            ]
        ).astype(np.float32)

    def _get_contact_state_for_env(self, env):
        left_contact = env.check_contact(
            env.robots[0].gripper.important_geoms["left_fingerpad"],
            env.cube,
        )
        right_contact = env.check_contact(
            env.robots[0].gripper.important_geoms["right_fingerpad"],
            env.cube,
        )
        return left_contact, right_contact, bool(left_contact or right_contact)

    def _get_contact_state(self):
        return self._get_contact_state_for_env(self.env)

    def _get_table_contact_state_for_env(self, env):
        left_table_contact = env.check_contact(
            env.robots[0].gripper.important_geoms["left_finger"],
            "table_collision",
        )
        right_table_contact = env.check_contact(
            env.robots[0].gripper.important_geoms["right_finger"],
            "table_collision",
        )
        return left_table_contact, right_table_contact, bool(left_table_contact or right_table_contact)

    def _get_table_contact_state(self):
        return self._get_table_contact_state_for_env(self.env)

    def _get_grasp_zone_state(self, obs):
        eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
        grasp_offset = np.array(self.object_profile["grasp_offset"], dtype=np.float64)
        grasp_goal = cube_pos + grasp_offset
        grasp_delta = grasp_goal - eef_pos
        xy_dist = float(np.linalg.norm(grasp_delta[:2]))
        vertical_error = float(grasp_delta[2])
        in_grasp_zone = (
            xy_dist < self.grasp_zone_xy_threshold
            and abs(vertical_error) < self.grasp_zone_z_threshold
        )
        return grasp_goal, grasp_delta, xy_dist, vertical_error, in_grasp_zone

    def _get_stage_state(self, obs, env=None):
        env = self.env if env is None else env
        grasp_goal, grasp_delta, grasp_xy_dist, vertical_error, in_grasp_zone = self._get_grasp_zone_state(obs)
        gripper_qpos = np.array(obs["robot0_gripper_qpos"], dtype=np.float64)
        left_contact, right_contact, any_contact = self._get_contact_state_for_env(env)
        _, _, any_table_contact = self._get_table_contact_state_for_env(env)
        grasped = env._check_grasp(env.robots[0].gripper, env.cube)

        bilateral_contact = bool(left_contact and right_contact)
        gripper_width = float(np.mean(np.abs(gripper_qpos)))
        gripper_closed = bool(gripper_width < 0.012)
        gripper_open = bool(gripper_width > 0.02)
        xy_aligned = bool(grasp_xy_dist < self.grasp_zone_xy_threshold * 1.35)
        height_aligned = bool(abs(vertical_error) < self.grasp_zone_z_threshold * 1.4)
        close_ready = bool(in_grasp_zone or any_contact or (xy_aligned and height_aligned))
        cage_stage = bool(xy_aligned and height_aligned and not grasped)

        return {
            "grasp_goal": grasp_goal,
            "grasp_delta": grasp_delta,
            "grasp_xy_dist": grasp_xy_dist,
            "vertical_error": vertical_error,
            "in_grasp_zone": in_grasp_zone,
            "xy_aligned": xy_aligned,
            "height_aligned": height_aligned,
            "close_ready": close_ready,
            "cage_stage": cage_stage,
            "left_contact": bool(left_contact),
            "right_contact": bool(right_contact),
            "any_contact": bool(any_contact),
            "bilateral_contact": bilateral_contact,
            "any_table_contact": bool(any_table_contact),
            "gripper_width": gripper_width,
            "gripper_closed": gripper_closed,
            "gripper_open": gripper_open,
            "grasped": bool(grasped),
        }

    def _apply_action_constraints(self, obs, action):
        adjusted_action = np.array(action, dtype=np.float32).copy()
        eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        stage = self._get_stage_state(obs)

        table_guard_active = False
        close_assist_active = False

        if eef_pos[2] < self.table_guard_height and adjusted_action[2] < 0:
            adjusted_action[2] = 0.0
            table_guard_active = True

        if stage["any_table_contact"] and adjusted_action[2] < 0.35:
            adjusted_action[2] = 0.35
            table_guard_active = True

        if stage["any_contact"] and adjusted_action[-1] < 0.7:
            adjusted_action[-1] = 0.7
            close_assist_active = True
        elif stage["close_ready"] and adjusted_action[-1] < 0.45:
            adjusted_action[-1] = 0.45
            close_assist_active = True

        return adjusted_action, table_guard_active, close_assist_active

    def _compute_reward(self, obs, action):
        eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
        stage = self._get_stage_state(obs)
        grasp_goal = stage["grasp_goal"]
        grasp_delta = stage["grasp_delta"]
        grasp_xy_dist = stage["grasp_xy_dist"]
        vertical_error = stage["vertical_error"]

        grasp_dist = float(np.linalg.norm(grasp_delta))
        ori_error = float(np.linalg.norm(self._quat_axis_error(self.current_target_quat, eef_quat)))
        cube_height = float(cube_pos[2] - self.table_height)
        lift_amount = max(float(cube_pos[2] - self.reset_cube_height), 0.0) if self.reset_cube_height is not None else 0.0
        gripper_cmd = float(action[-1]) if len(action) > 0 else 0.0
        gripper_closed = float(stage["gripper_closed"])
        gripper_open = float(stage["gripper_open"])
        any_contact = stage["any_contact"]
        any_table_contact = stage["any_table_contact"]
        bilateral_contact = stage["bilateral_contact"]
        grasped = stage["grasped"]
        success = lift_amount > self.object_profile["success_height"]
        table_clearance = float(eef_pos[2] - self.table_height)
        stage_reach = float(not stage["close_ready"])
        stage_cage = float(stage["cage_stage"] and not any_contact)
        stage_grasp = float(any_contact or grasped)
        prev_grasp_dist = self.prev_grasp_dist if self.prev_grasp_dist is not None else grasp_dist
        prev_grasp_xy_dist = self.prev_grasp_xy_dist if self.prev_grasp_xy_dist is not None else grasp_xy_dist
        prev_vertical_error_abs = (
            self.prev_vertical_error_abs if self.prev_vertical_error_abs is not None else abs(vertical_error)
        )
        prev_lift_amount = self.prev_lift_amount if self.prev_lift_amount is not None else lift_amount

        grasp_progress = prev_grasp_dist - grasp_dist
        xy_progress = prev_grasp_xy_dist - grasp_xy_dist
        vertical_progress = prev_vertical_error_abs - abs(vertical_error)
        lift_progress = lift_amount - prev_lift_amount

        reward_parts = {
            "reach_cube": (0.16 / (1.0 + 14.0 * grasp_dist)) if stage_reach else 0.0,
            "grasp_progress": 2.4 * grasp_progress,
            "xy_progress": 1.8 * xy_progress,
            "vertical_progress": 1.2 * vertical_progress if stage["xy_aligned"] else 0.0,
            "cage_xy": 0.28 / (1.0 + 20.0 * grasp_xy_dist),
            "cage_height": (0.22 / (1.0 + 25.0 * abs(vertical_error))) if stage["xy_aligned"] else 0.0,
            "orientation": 0.10 / (1.0 + 4.0 * ori_error),
            "open_approach_bonus": 0.12 if stage_reach and gripper_open else 0.0,
            "cage_bonus": 0.22 if stage["cage_stage"] else 0.0,
            "contact": 0.25 if any_contact else 0.0,
            "bilateral_contact": 0.40 if bilateral_contact else 0.0,
            "close_ready_bonus": 0.30 if stage["close_ready"] and gripper_cmd > 0 else 0.0,
            "contact_close_synergy": 0.25 if any_contact and gripper_cmd > 0 else 0.0,
            "hesitate_close_penalty": -0.20 if stage["cage_stage"] and gripper_cmd <= 0 and not any_contact else 0.0,
            "grasp": 1.00 if grasped else 0.0,
            "lift": lift_amount * (14.0 if grasped else 3.0 if bilateral_contact else 0.0),
            "lift_progress": 12.0 * lift_progress if stage_grasp else 0.0,
            "stable_grasp": 0.70 if grasped and lift_amount > 0.015 else 0.0,
            "success": 5.0 if success else 0.0,
            "early_close_penalty": -0.40 if gripper_cmd > 0 and not stage["close_ready"] else 0.0,
            "premature_closed_penalty": -0.35 if gripper_closed and not stage["close_ready"] else 0.0,
            "table_contact_penalty": -1.00 if any_table_contact else 0.0,
            "table_clearance_penalty": -0.20 if table_clearance < 0.03 else 0.0,
            "downward_near_table_penalty": -0.35 if table_clearance < 0.05 and action[2] < -0.05 else 0.0,
            "action_penalty": -0.01 * float(np.linalg.norm(action[:6])),
            "time_penalty": -0.005,
        }
        reward = sum(reward_parts.values())
        reward_parts["stage_reach"] = stage_reach
        reward_parts["stage_cage"] = stage_cage
        reward_parts["stage_grasp"] = stage_grasp
        reward_parts["close_ready"] = float(stage["close_ready"])
        reward_parts["in_grasp_zone"] = float(stage["in_grasp_zone"])
        reward_parts["xy_aligned"] = float(stage["xy_aligned"])
        reward_parts["height_aligned"] = float(stage["height_aligned"])
        reward_parts["bilateral_contact_flag"] = float(bilateral_contact)
        reward_parts["grasp_xy_dist"] = grasp_xy_dist
        reward_parts["vertical_error"] = vertical_error
        reward_parts["table_clearance"] = table_clearance
        reward_parts["lift_amount"] = lift_amount
        reward_parts["cube_height"] = cube_height
        reward_parts["grasp_dist"] = grasp_dist
        reward_parts["gripper_width"] = stage["gripper_width"]
        reward_parts["grasp_progress_value"] = grasp_progress
        reward_parts["xy_progress_value"] = xy_progress
        reward_parts["vertical_progress_value"] = vertical_progress
        reward_parts["lift_progress_value"] = lift_progress
        return reward, reward_parts, success

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.reset_count += 1
        self.stage_a_handoff_gate.reset()

        cube_pos = self._sample_cube_position()
        ik_result = self._solve_ik_pregrasp(cube_pos)
        if ik_result["success"]:
            self.reset_successes += 1
        else:
            self.reset_failures += 1
        return self._finalize_reset(
            cube_pos=cube_pos,
            arm_qpos=ik_result["arm_qpos"],
            target_pos=ik_result["target_pos"],
            target_quat=ik_result["target_quat"],
            reset_success=ik_result["success"],
            reset_attempts=ik_result["attempt"],
            reset_strategy="ik_pregrasp",
        )

    def reset_to_manual_start(self, seed=None, options=None):
        """Resets the episode context, then rewinds the arm to a farther manual-teleop start pose."""
        _, info = self.reset(seed=seed, options=options)
        cube_pos = np.array(info["cube_pos"], dtype=np.float64)

        self.stage_a_handoff_gate.reset()
        self.env.robots[0].init_qpos = self.default_arm_qpos.copy()
        self.env.reset()
        self._set_gripper_fully_open(self.env)
        self._set_free_joint_pose(self.env, self.cube_joint, cube_pos)
        self.env.sim.forward()

        raw_obs = self.env._get_observations(force_update=True)
        self.last_raw_obs = raw_obs
        self.prev_grasp_dist = None
        self.prev_grasp_xy_dist = None
        self.prev_vertical_error_abs = None
        self.prev_lift_amount = 0.0
        handoff_info = self.evaluate_stage_a_handoff(raw_obs, update_streak=False)
        self.last_reset_meta.update(
            {
                "manual_start": True,
                "can_handoff_to_stage_a": bool(handoff_info["can_handoff_to_stage_a"]),
                "stage_a_handoff_ready_now": bool(handoff_info["ready_now"]),
                "stage_a_handoff_consecutive_hits": int(handoff_info["consecutive_hits"]),
                "stage_a_handoff_failure_reasons": list(handoff_info["failure_reasons"]),
            }
        )
        return raw_obs, dict(self.last_reset_meta)

    def step(self, action):
        current_obs = self.last_raw_obs if self.last_raw_obs is not None else self.env._get_observations(force_update=True)
        adjusted_action, table_guard_active, close_assist_active = self._apply_action_constraints(current_obs, action)
        obs, _, done, info = self.env.step(adjusted_action)
        obs_vec = self.sync_from_raw_observation(obs, update_handoff_streak=True)
        handoff_info = self.last_stage_a_handoff
        reward, reward_parts, success = self._compute_reward(obs, adjusted_action)
        self.prev_grasp_dist = reward_parts["grasp_dist"]
        self.prev_grasp_xy_dist = reward_parts["grasp_xy_dist"]
        self.prev_vertical_error_abs = abs(reward_parts["vertical_error"])
        self.prev_lift_amount = reward_parts["lift_amount"]

        info = dict(info)
        info.update(reward_parts)
        info.update(self.last_reset_meta)
        info["success"] = success
        info["table_guard_active"] = table_guard_active
        info["close_assist_active"] = close_assist_active
        info["raw_gripper_action"] = float(action[-1]) if len(action) > 0 else 0.0
        info["adjusted_gripper_action"] = float(adjusted_action[-1]) if len(adjusted_action) > 0 else 0.0
        info["can_handoff_to_stage_a"] = bool(handoff_info["can_handoff_to_stage_a"])
        info["stage_a_handoff_ready_now"] = bool(handoff_info["ready_now"])
        info["stage_a_handoff_consecutive_hits"] = int(handoff_info["consecutive_hits"])
        info["stage_a_handoff_failure_reasons"] = list(handoff_info["failure_reasons"])
        info["camera_name"] = self.camera_name
        info["visual_feature_dim"] = self.policy_visual_feature_dim
        info["policy_uses_visual"] = bool(self.policy_uses_visual)
        info["vision_role"] = self.vision_role
        info.update(self.latest_camera_summary)
        terminated = bool(done or success)
        truncated = False
        return obs_vec, float(reward), terminated, truncated, info

    def render(self):
        self.env.render()

    def close(self):
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
        if hasattr(self, "reset_helper_env") and self.reset_helper_env is not None:
            self.reset_helper_env.close()
