"""Stage C environment: start from a grasped pose and learn to lift / hold."""

from __future__ import annotations

import numpy as np
import robosuite.utils.transform_utils as T

from core.object_profiles import get_object_profile
from core.stage_b_grasp_env import StageBGraspEnv


class StageCLiftEnv(StageBGraspEnv):
    """
    Stage C assumes Stage B has already achieved a stable grasp.

    Reset strategy:
    - reuse Stage B reset so the cube starts between the jaws
    - run a short scripted close routine until grasp is established
    - start RL from a grasped-but-not-lifted state
    """

    def __init__(self, *args, horizon: int = 100, **kwargs):
        profile_name = kwargs.get("object_profile_name", "cube_small")
        vision_context = kwargs.get("vision_context")
        if isinstance(vision_context, dict) and vision_context.get("object_profile_name"):
            profile_name = vision_context["object_profile_name"]
        profile = get_object_profile(profile_name)
        stage_cfg = profile.get("stage_targets", {}).get("stage_c", {})
        local_target = stage_cfg.get("local_target", [0.0, 0.0, 0.0])
        workspace_bounds = stage_cfg.get("workspace_bounds", {})

        self.stage_c_local_x_target = float(local_target[0])
        self.stage_c_local_y_target = float(local_target[1])
        self.stage_c_local_z_target = float(local_target[2])
        self.stage_c_reset_close_steps = int(stage_cfg.get("reset_close_steps", 24))
        self.stage_c_post_grasp_settle_steps = int(stage_cfg.get("post_grasp_settle_steps", 4))
        self.stage_c_reset_retry_limit = int(stage_cfg.get("reset_retry_limit", 4))
        self.stage_c_max_local_x = float(workspace_bounds.get("x", 0.08))
        self.stage_c_max_local_y = float(workspace_bounds.get("y", 0.05))
        self.stage_c_min_local_z = float(workspace_bounds.get("z_min", -0.03))
        self.stage_c_max_local_z = float(workspace_bounds.get("z_max", 0.06))
        self.stage_c_lift_success_steps = int(stage_cfg.get("lift_success_steps", 3))
        self.stage_c_hold_close_action = float(stage_cfg.get("hold_close_action", 1.0))
        self.stage_c_success_height = float(stage_cfg.get("success_height", profile.get("success_height", 0.05)))
        self.stage_c_post_grasp_xy_action_limit = float(stage_cfg.get("post_grasp_xy_action_limit", 0.10))
        self.stage_c_post_grasp_rot_action_limit = float(stage_cfg.get("post_grasp_rot_action_limit", 0.05))
        self.stage_c_hold_bonus = float(stage_cfg.get("hold_bonus", 0.45))
        self.stage_c_lift_amount_scale = float(stage_cfg.get("lift_amount_scale", 22.0))
        self.stage_c_lift_progress_scale = float(stage_cfg.get("lift_progress_scale", 28.0))
        self.stage_c_upright_bonus_scale = float(stage_cfg.get("upright_bonus_scale", 0.8))
        self.stage_c_tilt_penalty_scale = float(stage_cfg.get("tilt_penalty_scale", 1.4))
        self.stage_c_prelift_height = float(stage_cfg.get("prelift_height", 0.012))
        self.stage_c_no_lift_penalty = float(stage_cfg.get("no_lift_penalty", 0.35))
        self.stage_c_downward_after_grasp_penalty = float(stage_cfg.get("downward_after_grasp_penalty", 0.45))
        self.stage_c_stall_after_grasp_penalty = float(stage_cfg.get("stall_after_grasp_penalty", 0.30))
        self.stage_c_upright_success_cos = float(stage_cfg.get("upright_success_cos", 0.96))
        self.stage_c_severe_tilt_cos = float(stage_cfg.get("severe_tilt_cos", 0.85))
        self.cube_half_extent = float(profile.get("grasp_offset", [0.0, 0.0, 0.028])[2] - 0.007)
        self.consecutive_lift_success_steps = 0
        super().__init__(*args, horizon=horizon, **kwargs)

    def _cube_lowest_z(self, cube_pos, cube_quat):
        cube_rot = T.quat2mat(cube_quat)
        support = self.cube_half_extent * float(np.sum(np.abs(cube_rot[2, :])))
        return float(cube_pos[2] - support), cube_rot

    def _scripted_close_to_grasp(self, obs):
        close_steps = 0
        grasp_success = False
        rollout_obs = obs

        for close_steps in range(1, self.stage_c_reset_close_steps + 1):
            stage = self._get_stage_state(rollout_obs)
            frame = self._get_gripper_frame_state(rollout_obs)
            if stage["grasped"]:
                grasp_success = True
                break

            pos_world = (
                frame["x_axis"] * (frame["local_x"] - self.stage_b_local_x_target)
                + frame["y_axis"] * (frame["local_y"] - self.stage_b_local_y_target)
                + frame["z_axis"] * (frame["local_z"] - self.stage_b_local_z_target)
            )
            eef_quat = np.array(rollout_obs["robot0_eef_quat"], dtype=np.float64)
            ori_error = self._quat_axis_error(self.current_target_quat, eef_quat)

            action = np.zeros(self.env.action_dim, dtype=np.float32)
            action[:3] = np.clip(pos_world / 0.015, -1.0, 1.0)
            action[3:6] = np.clip(ori_error / 0.10, -1.0, 1.0)
            action[-1] = 1.0
            rollout_obs, _, _, _ = self.env.step(action)
            self.env.sim.forward()

        if not grasp_success:
            grasp_success = bool(self._get_stage_state(rollout_obs)["grasped"])
        elif self.stage_c_post_grasp_settle_steps > 0:
            # Once grasp is established, keep a stronger close command for a few steps.
            # This acts like a small extra squeeze so the lift phase starts from a firmer hold.
            for _ in range(self.stage_c_post_grasp_settle_steps):
                settle_action = np.zeros(self.env.action_dim, dtype=np.float32)
                settle_action[-1] = self.stage_c_hold_close_action
                rollout_obs, _, _, _ = self.env.step(settle_action)
                self.env.sim.forward()

        return rollout_obs, grasp_success, close_steps

    def _apply_action_constraints(self, obs, action):
        adjusted_action, table_guard_active, close_assist_active = super()._apply_action_constraints(obs, action)
        stage = self._get_stage_state(obs)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
        lift_amount = max(float(cube_pos[2] - self.reset_cube_height), 0.0) if self.reset_cube_height is not None else 0.0

        squeeze_assist_active = False
        if (stage["grasped"] or stage["bilateral_contact"]) and lift_amount < self.stage_c_success_height:
            if adjusted_action[-1] < self.stage_c_hold_close_action:
                adjusted_action[-1] = self.stage_c_hold_close_action
                squeeze_assist_active = True

        if stage["grasped"] or stage["bilateral_contact"]:
            adjusted_action[0] = float(np.clip(adjusted_action[0], -self.stage_c_post_grasp_xy_action_limit, self.stage_c_post_grasp_xy_action_limit))
            adjusted_action[1] = float(np.clip(adjusted_action[1], -self.stage_c_post_grasp_xy_action_limit, self.stage_c_post_grasp_xy_action_limit))
            adjusted_action[2] = float(np.clip(adjusted_action[2], 0.0, 1.0))
            adjusted_action[3] = float(np.clip(adjusted_action[3], -self.stage_c_post_grasp_rot_action_limit, self.stage_c_post_grasp_rot_action_limit))
            adjusted_action[4] = float(np.clip(adjusted_action[4], -self.stage_c_post_grasp_rot_action_limit, self.stage_c_post_grasp_rot_action_limit))
            adjusted_action[5] = float(np.clip(adjusted_action[5], -self.stage_c_post_grasp_rot_action_limit, self.stage_c_post_grasp_rot_action_limit))

        return adjusted_action, table_guard_active, close_assist_active, squeeze_assist_active

    def _build_observation(self, obs):
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float32)
        frame = self._get_gripper_frame_state(obs)
        stage = self._get_stage_state(obs)
        visual_features = self._build_visual_features(obs) if self.policy_uses_visual else None

        orientation_error = self._quat_axis_error(self.current_target_quat, eef_quat).astype(np.float32)
        gripper_opening = np.array([stage["gripper_width"]], dtype=np.float32)
        cube_height = np.array([cube_pos[2] - self.table_height], dtype=np.float32)
        lift_amount = np.array(
            [max(float(cube_pos[2] - self.reset_cube_height), 0.0) if self.reset_cube_height is not None else 0.0],
            dtype=np.float32,
        )
        table_clearance = np.array([frame["jaw_center"][2] - self.table_height], dtype=np.float32)
        contact_flags = np.array(
            [
                1.0 if stage["any_contact"] else 0.0,
                1.0 if stage["bilateral_contact"] else 0.0,
                1.0 if stage["grasped"] else 0.0,
            ],
            dtype=np.float32,
        )
        local_state = np.array(
            [
                frame["local_x"] - self.stage_c_local_x_target,
                frame["local_y"] - self.stage_c_local_y_target,
                frame["local_z"] - self.stage_c_local_z_target,
                frame["pad_gap"],
            ],
            dtype=np.float32,
        )

        parts = [
            local_state,
            orientation_error,
            gripper_opening,
            cube_height,
            lift_amount,
            table_clearance,
            contact_flags,
            self.condition_features,
        ]
        if visual_features is not None:
            parts.append(visual_features)
        return np.concatenate(parts).astype(np.float32)

    def _compute_reward(self, obs, action):
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
        cube_quat = np.array(obs["cube_quat"], dtype=np.float64)
        stage = self._get_stage_state(obs)
        frame = self._get_gripper_frame_state(obs)
        gripper_cmd = float(action[-1]) if len(action) > 0 else 0.0

        local_x = frame["local_x"] - self.stage_c_local_x_target
        local_y = frame["local_y"] - self.stage_c_local_y_target
        local_z_error = frame["local_z"] - self.stage_c_local_z_target
        local_dist = float(np.linalg.norm([local_x, local_y, local_z_error]))
        ori_error = float(np.linalg.norm(self._quat_axis_error(self.current_target_quat, eef_quat)))
        table_clearance = float(frame["jaw_center"][2] - self.table_height)
        lift_amount = max(float(cube_pos[2] - self.reset_cube_height), 0.0) if self.reset_cube_height is not None else 0.0
        cube_lowest_z, cube_rot = self._cube_lowest_z(cube_pos, cube_quat)
        bottom_lift_amount = (
            max(float(cube_lowest_z - self.reset_cube_lowest_z), 0.0) if self.reset_cube_lowest_z is not None else 0.0
        )
        upright_cos = float(cube_rot[2, 2])
        prev_lift_amount = self.prev_lift_amount if self.prev_lift_amount is not None else bottom_lift_amount
        lift_progress = bottom_lift_amount - prev_lift_amount

        prev_grasp_dist = self.prev_grasp_dist if self.prev_grasp_dist is not None else local_dist
        prev_grasp_xy_dist = self.prev_grasp_xy_dist if self.prev_grasp_xy_dist is not None else abs(local_y)
        prev_vertical_error_abs = (
            self.prev_vertical_error_abs if self.prev_vertical_error_abs is not None else abs(local_z_error)
        )
        progress = prev_grasp_dist - local_dist
        lateral_progress = prev_grasp_xy_dist - abs(local_y)
        vertical_progress = prev_vertical_error_abs - abs(local_z_error)
        not_lifting_enough = bool(stage["grasped"] and bottom_lift_amount < self.stage_c_prelift_height)
        stale_lift = bool(stage["grasped"] and lift_progress <= 1e-4 and bottom_lift_amount < self.stage_c_success_height)
        downward_after_grasp = bool(stage["grasped"] and action[2] < 0.02)
        severe_tilt = bool(stage["grasped"] and upright_cos < self.stage_c_severe_tilt_cos)

        lift_success = bool(
            stage["grasped"]
            and bottom_lift_amount > self.stage_c_success_height
            and upright_cos > self.stage_c_upright_success_cos
            and self.consecutive_lift_success_steps >= self.stage_c_lift_success_steps
            and not stage["any_table_contact"]
        )
        out_of_workspace = bool(
            abs(local_x) > self.stage_c_max_local_x
            or abs(local_y) > self.stage_c_max_local_y
            or frame["local_z"] < self.stage_c_min_local_z
            or frame["local_z"] > self.stage_c_max_local_z
        )

        reward_parts = {
            "distance_penalty": -1.8 * local_dist,
            "local_x_penalty": -3.5 * abs(local_x),
            "local_y_penalty": -4.5 * abs(local_y),
            "local_z_penalty": -2.2 * abs(local_z_error),
            "orientation_penalty": -0.05 * ori_error,
            "grasp_hold_bonus": self.stage_c_hold_bonus if stage["grasped"] else -1.00,
            "bilateral_contact_bonus": 0.50 if stage["bilateral_contact"] else 0.0,
            "lift_amount_bonus": self.stage_c_lift_amount_scale * bottom_lift_amount if stage["grasped"] else 0.0,
            "lift_progress_bonus": self.stage_c_lift_progress_scale * lift_progress if stage["grasped"] else 0.0,
            "upright_bonus": self.stage_c_upright_bonus_scale * max(upright_cos - self.stage_c_severe_tilt_cos, 0.0)
            if stage["grasped"]
            else 0.0,
            "tilt_penalty": -self.stage_c_tilt_penalty_scale * max(self.stage_c_upright_success_cos - upright_cos, 0.0)
            if stage["grasped"]
            else 0.0,
            "maintain_center_bonus": 0.30 if abs(local_y) < self.stage_b_local_y_success_threshold else 0.0,
            "maintain_height_bonus": 0.20 if abs(local_z_error) < 0.02 else 0.0,
            "progress": 1.0 * progress,
            "lateral_progress": 1.5 * lateral_progress,
            "vertical_progress": 1.0 * vertical_progress,
            "upward_bonus": 0.35 if action[2] > 0.05 and stage["grasped"] else 0.0,
            "premature_close_penalty": -0.10 if gripper_cmd < 0.2 else 0.0,
            "lost_grasp_penalty": -1.20 if not stage["grasped"] else 0.0,
            "no_lift_after_grasp_penalty": -self.stage_c_no_lift_penalty if not_lifting_enough else 0.0,
            "downward_after_grasp_penalty": -self.stage_c_downward_after_grasp_penalty if downward_after_grasp else 0.0,
            "stall_after_grasp_penalty": -self.stage_c_stall_after_grasp_penalty if stale_lift else 0.0,
            "severe_tilt_penalty": -0.60 if severe_tilt else 0.0,
            "table_contact_penalty": -1.20 if stage["any_table_contact"] else 0.0,
            "workspace_escape_penalty": -1.00 if out_of_workspace else 0.0,
            "action_penalty": -0.01 * float(np.linalg.norm(action[:6])),
            "time_penalty": -0.004,
            "success": 5.0 if lift_success else 0.0,
        }

        reward = sum(reward_parts.values())
        reward_parts["success_flag"] = float(lift_success)
        reward_parts["local_x"] = local_x
        reward_parts["local_y"] = local_y
        reward_parts["local_z"] = frame["local_z"]
        reward_parts["local_z_error"] = local_z_error
        reward_parts["pad_gap"] = frame["pad_gap"]
        reward_parts["table_clearance"] = table_clearance
        reward_parts["any_contact"] = float(stage["any_contact"])
        reward_parts["bilateral_contact"] = float(stage["bilateral_contact"])
        reward_parts["grasped"] = float(stage["grasped"])
        reward_parts["any_table_contact"] = float(stage["any_table_contact"])
        reward_parts["gripper_width"] = stage["gripper_width"]
        reward_parts["out_of_workspace"] = float(out_of_workspace)
        reward_parts["lift_amount"] = lift_amount
        reward_parts["bottom_lift_amount"] = bottom_lift_amount
        reward_parts["lift_progress_value"] = lift_progress
        reward_parts["grasp_progress_value"] = progress
        reward_parts["xy_progress_value"] = lateral_progress
        reward_parts["vertical_progress_value"] = vertical_progress
        reward_parts["upright_cos"] = upright_cos
        reward_parts["not_lifting_enough"] = float(not_lifting_enough)
        reward_parts["stale_lift"] = float(stale_lift)
        reward_parts["downward_after_grasp"] = float(downward_after_grasp)
        reward_parts["severe_tilt"] = float(severe_tilt)
        return reward, reward_parts, lift_success

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        base_info = {}
        grasp_ready = False
        close_steps = 0

        for attempt in range(1, self.stage_c_reset_retry_limit + 1):
            _, base_info = super().reset(seed=seed if attempt == 1 else None, options=options)
            obs = self.last_raw_obs
            obs, grasp_ready, close_steps = self._scripted_close_to_grasp(obs)
            self.last_raw_obs = obs
            if grasp_ready:
                break

        self.current_target_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        self.reset_cube_height = float(obs["cube_pos"][2])
        self.reset_cube_lowest_z, _ = self._cube_lowest_z(
            np.array(obs["cube_pos"], dtype=np.float64),
            np.array(obs["cube_quat"], dtype=np.float64),
        )
        frame = self._get_gripper_frame_state(obs)
        local_dist = float(
            np.linalg.norm(
                [
                    frame["local_x"] - self.stage_c_local_x_target,
                    frame["local_y"] - self.stage_c_local_y_target,
                    frame["local_z"] - self.stage_c_local_z_target,
                ]
            )
        )
        self.prev_grasp_dist = local_dist
        self.prev_grasp_xy_dist = float(abs(frame["local_y"] - self.stage_c_local_y_target))
        self.prev_vertical_error_abs = float(abs(frame["local_z"] - self.stage_c_local_z_target))
        self.prev_lift_amount = 0.0
        self.consecutive_grasp_steps = self.stage_b_grasp_success_steps if grasp_ready else 0
        self.consecutive_lift_success_steps = 0

        obs_vec = self._build_observation(obs)
        self.last_reset_meta = dict(base_info)
        self.last_reset_meta.update(
            {
                "pregrasp_success": bool(grasp_ready),
                "stage_c_start_success": bool(grasp_ready),
                "stage_c_close_steps": close_steps,
                "reset_helper_mode": f"{self.reset_helper_mode}+stage_b_grasp_script",
                "stage_c_local_x": frame["local_x"],
                "stage_c_local_y": frame["local_y"],
                "stage_c_local_z": frame["local_z"],
                "stage_c_pad_gap": frame["pad_gap"],
            }
        )
        return obs_vec, dict(self.last_reset_meta)

    def step(self, action):
        current_obs = self.last_raw_obs if self.last_raw_obs is not None else self.env._get_observations(force_update=True)
        adjusted_action, table_guard_active, close_assist_active, squeeze_assist_active = self._apply_action_constraints(
            current_obs, action
        )
        obs, _, done, info = self.env.step(adjusted_action)
        self.last_raw_obs = obs

        stage = self._get_stage_state(obs)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
        lift_amount = max(float(cube_pos[2] - self.reset_cube_height), 0.0) if self.reset_cube_height is not None else 0.0
        cube_lowest_z, _ = self._cube_lowest_z(cube_pos, np.array(obs["cube_quat"], dtype=np.float64))
        bottom_lift_amount = (
            max(float(cube_lowest_z - self.reset_cube_lowest_z), 0.0) if self.reset_cube_lowest_z is not None else 0.0
        )
        if stage["grasped"] and bottom_lift_amount > self.stage_c_success_height:
            self.consecutive_lift_success_steps += 1
        else:
            self.consecutive_lift_success_steps = 0

        obs_vec = self._build_observation(obs)
        reward, reward_parts, success = self._compute_reward(obs, adjusted_action)
        self.prev_grasp_dist = float(
            np.linalg.norm(
                [
                    reward_parts["local_x"],
                    reward_parts["local_y"],
                    reward_parts["local_z_error"],
                ]
            )
        )
        self.prev_grasp_xy_dist = abs(reward_parts["local_y"])
        self.prev_vertical_error_abs = abs(reward_parts["local_z_error"])
        self.prev_lift_amount = reward_parts["bottom_lift_amount"]

        info = dict(info)
        info.update(reward_parts)
        info.update(self.last_reset_meta)
        info["success"] = success
        info["table_guard_active"] = table_guard_active
        info["close_assist_active"] = close_assist_active
        info["squeeze_assist_active"] = squeeze_assist_active
        info["lift_success_streak"] = self.consecutive_lift_success_steps
        info["raw_gripper_action"] = float(action[-1]) if len(action) > 0 else 0.0
        info["adjusted_gripper_action"] = float(adjusted_action[-1]) if len(adjusted_action) > 0 else 0.0

        failure = bool(reward_parts["out_of_workspace"] > 0.0 or reward_parts["any_table_contact"] > 0.0)
        terminated = bool(done or success or failure)
        truncated = False
        return obs_vec, float(reward), terminated, truncated, info
