"""Stage A environment: cage the object without touching the table."""

from __future__ import annotations

import numpy as np
import robosuite.utils.transform_utils as T

from core.object_profiles import get_object_profile
from core.stage2_local_grasp_env import Stage2LocalGraspEnv
from core.vision.depth_features import encode_wrist_rgbd


class StageACageEnv(Stage2LocalGraspEnv):
    """
    Stage A only learns local alignment / caging.

    Behavior goals:
    - move the gripper into a clean pre-grasp cage pose around the cube
    - keep the gripper open
    - avoid table contact

    It explicitly does NOT try to learn closing or lifting yet.
    """

    def __init__(self, *args, horizon: int = 140, **kwargs):
        profile_name = kwargs.get("object_profile_name", "cube_small")
        vision_context = kwargs.get("vision_context")
        if isinstance(vision_context, dict) and vision_context.get("object_profile_name"):
            profile_name = vision_context["object_profile_name"]
        profile = get_object_profile(profile_name)
        stage_cfg = profile.get("stage_targets", {}).get("stage_a", {})
        reset_cfg = stage_cfg.get("teleop_handoff_reset", {})
        reset_strategy = kwargs.pop("reset_strategy", None)
        self.stage_a_workflow_mode = str(stage_cfg.get("workflow_mode", "cage_align"))
        local_target = stage_cfg.get("local_target", [0.0, 0.0, 0.0])
        thresholds = stage_cfg.get("success_thresholds", {})
        workspace_bounds = stage_cfg.get("workspace_bounds", {})
        descent_cfg = stage_cfg.get("descent_control", {})
        cage_geometry_cfg = stage_cfg.get("cage_geometry", {})
        visual_reference_cfg = stage_cfg.get("visual_reference", {})

        self.stage_a_local_x_target = float(local_target[0])
        self.stage_a_local_y_target = float(local_target[1])
        self.stage_a_local_z_target = float(local_target[2])
        self.stage_a_local_x_success_threshold = float(thresholds.get("x", 0.015))
        self.stage_a_local_y_success_threshold = float(thresholds.get("y", 0.007))
        self.stage_a_local_z_tolerance = float(thresholds.get("z", 0.014))
        self.stage_a_table_clearance_threshold = float(stage_cfg.get("table_clearance_threshold", 0.018))
        self.stage_a_open_width_threshold = float(stage_cfg.get("open_width_threshold", 0.03))
        self.stage_a_max_local_x = float(workspace_bounds.get("x", 0.10))
        self.stage_a_max_local_y = float(workspace_bounds.get("y", 0.06))
        self.stage_a_min_local_z = float(workspace_bounds.get("z_min", -0.035))
        self.stage_a_max_local_z = float(workspace_bounds.get("z_max", 0.035))
        self.stage_a_min_episode_steps = int(stage_cfg.get("min_episode_steps", 12))
        self.stage_a_failure_patience_steps = int(stage_cfg.get("failure_patience_steps", 6))
        self.stage_a_b_handoff_stable_steps = int(stage_cfg.get("b_handoff_stable_steps", 3))
        self.stage_a_descent_unlock_x = float(descent_cfg.get("x_unlock", 0.018))
        self.stage_a_descent_unlock_y = float(descent_cfg.get("y_unlock", 0.012))
        self.stage_a_prealign_downward_limit = float(descent_cfg.get("prealign_downward_limit", 0.01))
        self.stage_a_forced_descend_action = float(descent_cfg.get("forced_descend_action", -0.18))
        self.stage_a_object_half_extents = np.array(
            profile.get("object_half_extents", [0.02095668, 0.02134781, 0.02039102]),
            dtype=np.float64,
        )
        self.stage_a_fore_aft_cage_target = float(cage_geometry_cfg.get("fore_aft_target_x", self.stage_a_local_x_target))
        self.stage_a_fore_aft_cage_tolerance = float(cage_geometry_cfg.get("fore_aft_tolerance", self.stage_a_local_x_success_threshold))
        self.stage_a_lateral_cage_margin = float(cage_geometry_cfg.get("lateral_margin", 0.004))
        self.stage_a_visual_reference_enabled = bool(visual_reference_cfg.get("enabled", True))
        self.stage_a_visual_reference_progress_scale = float(visual_reference_cfg.get("progress_scale", 1.2))
        self.stage_a_visual_reference_error_scale = float(visual_reference_cfg.get("error_scale", 0.55))
        self.stage_a_visual_reference_match_bonus = float(visual_reference_cfg.get("match_bonus", 0.20))
        self.stage_a_visual_reference_match_threshold = float(visual_reference_cfg.get("match_threshold", 0.070))
        visual_reference_target = visual_reference_cfg.get("reference_local_target", {})
        self.stage_a_visual_reference_local_x_target = float(visual_reference_target.get("x", 0.0))
        self.stage_a_visual_reference_local_y_target = float(visual_reference_target.get("y", 0.0))
        self.stage_a_visual_reference_local_z_target = float(visual_reference_target.get("z", 0.012))
        self.stage_a_visual_reference_max_contact_retries = int(visual_reference_cfg.get("max_contact_retries", 4))
        self.stage_a_reset_strategy = str(reset_strategy or stage_cfg.get("reset_strategy", "teleop_like"))
        self.stage_a_reset_bank_size = int(reset_cfg.get("bank_size", 6))
        self.stage_a_reset_reuse_limit = int(reset_cfg.get("reuse_limit", 256))
        self.stage_a_reset_cube_refresh_threshold = float(reset_cfg.get("cube_refresh_threshold", 0.03))
        self.stage_a_reset_bank_build_attempt_limit = int(reset_cfg.get("bank_build_attempt_limit", 18))
        self.stage_a_reset_refine_steps = int(reset_cfg.get("refine_steps", 18))
        self.stage_a_reset_handoff_xy_threshold = float(reset_cfg.get("handoff_xy_threshold", 0.022))
        self.stage_a_reset_handoff_vertical_threshold = float(reset_cfg.get("handoff_vertical_threshold", 0.024))
        view_local_target = reset_cfg.get("view_local_target", {})
        view_local_tolerance = reset_cfg.get("view_local_tolerance", {})
        self.stage_a_reset_view_local_x_target = float(view_local_target.get("x", 0.015))
        self.stage_a_reset_view_local_y_target = float(view_local_target.get("y", 0.0))
        self.stage_a_reset_view_local_z_target = float(view_local_target.get("z", 0.055))
        self.stage_a_reset_view_local_x_tolerance = float(view_local_tolerance.get("x", 0.015))
        self.stage_a_reset_view_local_y_tolerance = float(view_local_tolerance.get("y", 0.015))
        self.stage_a_reset_view_local_z_tolerance = float(view_local_tolerance.get("z", 0.020))
        joint_noise_std = np.array(
            reset_cfg.get("joint_noise_std", [0.006, 0.010, 0.006, 0.012, 0.006, 0.012, 0.006]),
            dtype=np.float64,
        )
        if joint_noise_std.ndim == 0:
            joint_noise_std = np.full(7, float(joint_noise_std), dtype=np.float64)
        self.stage_a_reset_joint_noise_std = joint_noise_std.astype(np.float64)
        self.stage_a_reset_bank = []
        self.stage_a_reset_bank_cube_pos = None
        self.stage_a_reset_reuse_count = 0
        # Stage A only needs local translation. Keeping wrist rotations active makes
        # the task harder and slows down continuation from the already-good A policy.
        self.stage_a_xy_action_limit = float(stage_cfg.get("xy_action_limit", 0.08))
        self.stage_a_z_action_limit = float(stage_cfg.get("z_action_limit", 0.10))
        self.stage_a_step_count = 0
        self.stage_a_failure_streak = 0
        self.stage_a_b_handoff_streak = 0
        self.stage_a_reference_visual_feature = None
        self.stage_a_reference_visual_cube_pos = None
        self.stage_a_reference_visual_error = None
        self.stage_a_reference_camera_summary = {}
        super().__init__(*args, horizon=horizon, **kwargs)
        self.stage_a_visual_reference_enabled = bool(self.stage_a_visual_reference_enabled and self.policy_uses_visual)

    def _should_refresh_stage_a_reset_bank(self, cube_pos):
        if not self.stage_a_reset_bank:
            return True
        if self.stage_a_reset_reuse_count >= self.stage_a_reset_reuse_limit:
            return True
        if self.stage_a_reset_bank_cube_pos is None:
            return True
        cube_shift = float(np.linalg.norm(np.array(cube_pos, dtype=np.float64) - self.stage_a_reset_bank_cube_pos))
        return cube_shift > self.stage_a_reset_cube_refresh_threshold

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

    def _rebuild_stage_a_reset_bank(self, cube_pos):
        bank = []
        attempts_used = 0
        successes = 0
        filter_passes = 0
        while len(bank) < self.stage_a_reset_bank_size and attempts_used < self.stage_a_reset_bank_build_attempt_limit:
            if self.stage_a_workflow_mode == "topdown_descend":
                ik_result = self._solve_ik_topdown_start(cube_pos)
            else:
                ik_result = self._solve_ik_pregrasp(cube_pos)
            attempts_used += int(ik_result["attempt"])
            if not ik_result["success"]:
                continue
            successes += 1
            helper_obs = self._refine_stage_a_teleop_like_pose(cube_pos)
            if not self._is_stage_a_teleop_like_candidate(helper_obs, env=self.reset_helper_env):
                continue
            filter_passes += 1
            bank.append(
                {
                    "arm_qpos": np.array(
                        self.reset_helper_env.sim.data.qpos[self.reset_helper_env.robots[0]._ref_joint_pos_indexes],
                        dtype=np.float64,
                    ),
                    "target_pos": np.array(helper_obs["robot0_eef_pos"], dtype=np.float64),
                    "target_quat": np.array(helper_obs["robot0_eef_quat"], dtype=np.float64),
                }
            )

        self.stage_a_reset_bank = bank
        self.stage_a_reset_bank_cube_pos = np.array(cube_pos, dtype=np.float64)
        self.stage_a_reset_reuse_count = 0
        return {
            "successes": successes,
            "attempts_used": attempts_used,
            "bank_size": len(bank),
            "filter_passes": filter_passes,
        }

    def _sample_stage_a_reset_state(self):
        if not self.stage_a_reset_bank:
            raise RuntimeError("Stage A teleop-like reset bank is empty.")

        sample_index = int(self.rng.integers(len(self.stage_a_reset_bank)))
        sample = self.stage_a_reset_bank[sample_index]
        arm_qpos = np.array(sample["arm_qpos"], dtype=np.float64, copy=True)

        noise_std = self.stage_a_reset_joint_noise_std
        if noise_std.shape[0] != arm_qpos.shape[0]:
            noise_std = np.resize(noise_std, arm_qpos.shape[0])
        if self.stage_a_workflow_mode == "topdown_descend":
            # Top-down starts should stay nearly centered above the object.
            # Large joint perturbations reintroduce lateral drift and break the
            # whole point of the human-aligned-above-object workflow.
            noise_std = 0.15 * noise_std
        arm_qpos += self.rng.normal(loc=0.0, scale=noise_std, size=arm_qpos.shape[0])

        self.stage_a_reset_reuse_count += 1
        return {
            "arm_qpos": arm_qpos,
            "target_pos": np.array(sample["target_pos"], dtype=np.float64, copy=True),
            "target_quat": np.array(sample["target_quat"], dtype=np.float64, copy=True),
            "sample_index": sample_index,
        }

    def _get_gripper_frame_state(self, obs, env=None):
        env = self.env if env is None else env
        sim = env.sim
        gripper = env.robots[0].gripper
        site_name = gripper.important_sites["grip_site"]
        ee_x = gripper.important_sites["ee_x"]
        ee_y = gripper.important_sites["ee_y"]
        ee_z = gripper.important_sites["ee_z"]

        grip_site = np.array(sim.data.get_site_xpos(site_name), dtype=np.float64)
        x_axis = np.array(sim.data.get_site_xpos(ee_x), dtype=np.float64) - grip_site
        y_axis = np.array(sim.data.get_site_xpos(ee_y), dtype=np.float64) - grip_site
        z_axis = np.array(sim.data.get_site_xpos(ee_z), dtype=np.float64) - grip_site
        x_axis /= np.linalg.norm(x_axis)
        y_axis /= np.linalg.norm(y_axis)
        z_axis /= np.linalg.norm(z_axis)

        left_pad = env.robots[0].gripper.important_geoms["left_fingerpad"][0]
        right_pad = env.robots[0].gripper.important_geoms["right_fingerpad"][0]
        left_id = sim.model.geom_name2id(left_pad)
        right_id = sim.model.geom_name2id(right_pad)
        left_pos = np.array(sim.data.geom_xpos[left_id], dtype=np.float64)
        right_pos = np.array(sim.data.geom_xpos[right_id], dtype=np.float64)
        pad_gap = float(np.linalg.norm(right_pos - left_pos))
        jaw_center = 0.5 * (left_pos + right_pos)

        cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
        cube_quat = np.array(obs["cube_quat"], dtype=np.float64)
        rel = cube_pos - jaw_center
        local_x = float(np.dot(rel, x_axis))
        local_y = float(np.dot(rel, y_axis))
        local_z = float(np.dot(rel, z_axis))
        planar_rel = rel.copy()
        planar_rel[2] = 0.0
        planar_y_axis = (right_pos - left_pos).astype(np.float64)
        planar_y_axis[2] = 0.0
        planar_y_norm = float(np.linalg.norm(planar_y_axis))
        if planar_y_norm < 1.0e-6:
            planar_y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        else:
            planar_y_axis /= planar_y_norm
        planar_x_axis = x_axis.copy()
        planar_x_axis[2] = 0.0
        planar_x_axis = planar_x_axis - float(np.dot(planar_x_axis, planar_y_axis)) * planar_y_axis
        planar_x_norm = float(np.linalg.norm(planar_x_axis))
        if planar_x_norm < 1.0e-6:
            planar_x_axis = np.array([-planar_y_axis[1], planar_y_axis[0], 0.0], dtype=np.float64)
        else:
            planar_x_axis /= planar_x_norm
        planar_local_x = float(np.dot(planar_rel, planar_x_axis))
        planar_local_y = float(np.dot(planar_rel, planar_y_axis))
        cube_rot = T.quat2mat(cube_quat)
        cube_axis_x = cube_rot[:, 0].copy()
        cube_axis_y = cube_rot[:, 1].copy()
        cube_axis_x[2] = 0.0
        cube_axis_y[2] = 0.0
        cube_axis_x /= max(np.linalg.norm(cube_axis_x), 1.0e-6)
        cube_axis_y /= max(np.linalg.norm(cube_axis_y), 1.0e-6)
        support_x = float(
            abs(np.dot(x_axis, cube_rot[:, 0])) * self.stage_a_object_half_extents[0]
            + abs(np.dot(x_axis, cube_rot[:, 1])) * self.stage_a_object_half_extents[1]
            + abs(np.dot(x_axis, cube_rot[:, 2])) * self.stage_a_object_half_extents[2]
        )
        support_y = float(
            abs(np.dot(y_axis, cube_rot[:, 0])) * self.stage_a_object_half_extents[0]
            + abs(np.dot(y_axis, cube_rot[:, 1])) * self.stage_a_object_half_extents[1]
            + abs(np.dot(y_axis, cube_rot[:, 2])) * self.stage_a_object_half_extents[2]
        )
        lateral_half_gap = 0.5 * pad_gap
        lateral_cage_limit = max(0.0, lateral_half_gap - support_y + self.stage_a_lateral_cage_margin)
        laterally_caged = bool(abs(planar_local_y - self.stage_a_local_y_target) <= lateral_cage_limit)
        desired_x_min = self.stage_a_fore_aft_cage_target - self.stage_a_fore_aft_cage_tolerance
        desired_x_max = self.stage_a_fore_aft_cage_target + self.stage_a_fore_aft_cage_tolerance
        object_x_min = local_x - support_x
        object_x_max = local_x + support_x
        fore_aft_caged = bool(object_x_min <= desired_x_min and object_x_max >= desired_x_max)
        cage_enclosed = bool(laterally_caged and fore_aft_caged)

        centered_between_fingers = bool(abs(local_y - self.stage_a_local_y_target) < self.stage_a_local_y_success_threshold)
        fore_aft_aligned = bool(abs(local_x - self.stage_a_local_x_target) < self.stage_a_local_x_success_threshold)
        height_aligned = bool(abs(local_z - self.stage_a_local_z_target) < self.stage_a_local_z_tolerance)
        open_enough = bool(float(np.mean(np.abs(obs["robot0_gripper_qpos"]))) > self.stage_a_open_width_threshold)

        return {
            "grip_site": grip_site,
            "jaw_center": jaw_center,
            "left_pad": left_pos,
            "right_pad": right_pos,
            "x_axis": x_axis,
            "y_axis": y_axis,
            "z_axis": z_axis,
            "local_x": local_x,
            "local_y": local_y,
            "local_z": local_z,
            "planar_local_x": planar_local_x,
            "planar_local_y": planar_local_y,
            "pad_gap": pad_gap,
            "cube_support_x": support_x,
            "cube_support_y": support_y,
            "lateral_half_gap": lateral_half_gap,
            "lateral_cage_limit": lateral_cage_limit,
            "object_x_min": object_x_min,
            "object_x_max": object_x_max,
            "laterally_caged": laterally_caged,
            "fore_aft_caged": fore_aft_caged,
            "cage_enclosed": cage_enclosed,
            "centered_between_fingers": centered_between_fingers,
            "fore_aft_aligned": fore_aft_aligned,
            "height_aligned": height_aligned,
            "open_enough": open_enough,
        }

    def _refine_stage_a_teleop_like_pose(self, cube_pos):
        helper_env = self.reset_helper_env
        obs = helper_env._get_observations(force_update=True)
        target_quat = self.default_controller_orientation.copy()

        for _ in range(self.stage_a_reset_refine_steps):
            frame = self._get_gripper_frame_state(obs, env=helper_env)
            local_x_error = frame["local_x"] - self.stage_a_reset_view_local_x_target
            local_y_error = frame["local_y"] - self.stage_a_reset_view_local_y_target
            local_z_error = frame["local_z"] - self.stage_a_reset_view_local_z_target
            if (
                abs(local_x_error) <= self.stage_a_reset_view_local_x_tolerance
                and abs(local_y_error) <= self.stage_a_reset_view_local_y_tolerance
                and abs(local_z_error) <= self.stage_a_reset_view_local_z_tolerance
            ):
                break

            eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
            ori_error = self._quat_axis_error(target_quat, eef_quat)
            action = np.zeros(helper_env.action_dim, dtype=np.float32)
            pos_world = (
                frame["x_axis"] * local_x_error
                + frame["y_axis"] * local_y_error
                + frame["z_axis"] * local_z_error
            )
            action[:3] = np.clip(pos_world / 0.02, -0.8, 0.8)
            action[3:6] = np.clip(ori_error / 0.10, -0.5, 0.5)
            action[-1] = -1.0
            obs, _, _, _ = helper_env.step(action)

        return obs

    def _refresh_stage_a_visual_reference(self, cube_pos):
        if not self.stage_a_visual_reference_enabled:
            self.stage_a_reference_visual_feature = None
            self.stage_a_reference_visual_cube_pos = None
            self.stage_a_reference_camera_summary = {}
            return

        cube_pos = np.array(cube_pos, dtype=np.float64)
        if (
            self.stage_a_reference_visual_feature is not None
            and self.stage_a_reference_visual_cube_pos is not None
            and np.linalg.norm(cube_pos - self.stage_a_reference_visual_cube_pos) <= self.stage_a_reset_cube_refresh_threshold
        ):
            return

        helper_env = self.reset_helper_env
        helper_env.robots[0].init_qpos = self.default_arm_qpos.copy()
        obs = helper_env.reset()
        self._set_free_joint_pose(helper_env, self.reset_helper_cube_joint, cube_pos)
        helper_env.sim.forward()
        obs = helper_env._get_observations(force_update=True)

        if self.default_controller_orientation is None:
            self.default_controller_orientation = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        target_quat = self.default_controller_orientation.copy()
        contact_retries = 0

        for _ in range(max(self.stage_a_reset_refine_steps, 1)):
            frame = self._get_gripper_frame_state(obs, env=helper_env)
            local_x_error = frame["local_x"] - self.stage_a_visual_reference_local_x_target
            local_y_error = frame["local_y"] - self.stage_a_visual_reference_local_y_target
            local_z_error = frame["local_z"] - self.stage_a_visual_reference_local_z_target
            _, _, any_contact = self._get_contact_state_for_env(helper_env)

            if any_contact and contact_retries < self.stage_a_visual_reference_max_contact_retries:
                action = np.zeros(helper_env.action_dim, dtype=np.float32)
                action[2] = 0.30
                action[-1] = -1.0
                obs, _, _, _ = helper_env.step(action)
                contact_retries += 1
                continue

            if (
                frame["laterally_caged"]
                and frame["fore_aft_caged"]
                and abs(local_z_error) <= self.stage_a_local_z_tolerance
                and not any_contact
            ):
                break

            eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
            ori_error = self._quat_axis_error(target_quat, eef_quat)
            action = np.zeros(helper_env.action_dim, dtype=np.float32)
            pos_world = (
                frame["x_axis"] * local_x_error
                + frame["y_axis"] * local_y_error
                + frame["z_axis"] * local_z_error
            )
            action[:3] = np.clip(pos_world / 0.02, -0.6, 0.6)
            action[3:6] = np.clip(ori_error / 0.10, -0.4, 0.4)
            action[-1] = -1.0
            obs, _, _, _ = helper_env.step(action)

        camera_obs = self._get_camera_observation(obs, sim=helper_env.sim)
        visual_vec, _ = encode_wrist_rgbd(
            rgb=camera_obs.get("rgb"),
            depth_m=camera_obs.get("depth_m"),
            min_distance_m=self.depth_min_distance_m,
            max_distance_m=self.depth_max_distance_m,
            depth_shape=self.visual_depth_shape,
            rgb_shape=self.visual_rgb_shape,
            crop_fraction=self.visual_crop_fraction,
        )
        self.stage_a_reference_visual_feature = visual_vec.astype(np.float32)
        self.stage_a_reference_visual_cube_pos = cube_pos.copy()
        self.stage_a_reference_camera_summary = dict(self.latest_camera_summary)

    def _get_visual_reference_error(self):
        if self.stage_a_reference_visual_feature is None:
            return None
        current = np.asarray(self.latest_visual_feature_vector, dtype=np.float32)
        reference = np.asarray(self.stage_a_reference_visual_feature, dtype=np.float32)
        return float(np.mean(np.abs(current - reference)))

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

    def _is_stage_a_teleop_like_candidate(self, obs, env=None):
        env = self.env if env is None else env
        stage = self._get_stage_state(obs, env=env)
        frame = self._get_gripper_frame_state(obs, env=env)
        _, _, any_table_contact = self._get_table_contact_state_for_env(env)
        return bool(
            abs(frame["local_x"] - self.stage_a_reset_view_local_x_target) <= self.stage_a_reset_view_local_x_tolerance
            and abs(frame["local_y"] - self.stage_a_reset_view_local_y_target) <= self.stage_a_reset_view_local_y_tolerance
            and abs(frame["local_z"] - self.stage_a_reset_view_local_z_target) <= self.stage_a_reset_view_local_z_tolerance
            and stage["gripper_open"]
            and not stage["any_contact"]
            and not any_table_contact
        )

    def _build_observation(self, obs):
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float32)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float32)
        gripper_qpos = np.array(obs["robot0_gripper_qpos"], dtype=np.float32)
        frame = self._get_gripper_frame_state(obs)
        visual_features = self._build_visual_features(obs) if self.policy_uses_visual else None

        orientation_error = self._quat_axis_error(self.current_target_quat, eef_quat).astype(np.float32)
        gripper_opening = np.array([float(np.mean(np.abs(gripper_qpos)))], dtype=np.float32)
        cube_height = np.array([cube_pos[2] - self.table_height], dtype=np.float32)
        table_clearance = np.array([frame["jaw_center"][2] - self.table_height], dtype=np.float32)
        _, _, any_table_contact = self._get_table_contact_state()
        table_contact_flag = np.array([1.0 if any_table_contact else 0.0], dtype=np.float32)
        local_state = np.array(
            [
                frame["local_x"],
                frame["local_y"] - self.stage_a_local_y_target,
                frame["local_z"] - self.stage_a_local_z_target,
                frame["pad_gap"],
            ],
            dtype=np.float32,
        )

        parts = [
            local_state,
            orientation_error,
            gripper_opening,
            cube_height,
            table_clearance,
            table_contact_flag,
            self.condition_features,
        ]
        if visual_features is not None:
            parts.append(visual_features)
        return np.concatenate(parts).astype(np.float32)

    def _build_scripted_topdown_action(self, obs):
        frame = self._get_gripper_frame_state(obs)
        eef_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
        cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
        eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        xy_error_world = cube_pos[:2] - eef_pos[:2]
        xy_ready = bool(
            abs(float(xy_error_world[0])) <= self.stage_a_descent_unlock_x
            and abs(float(xy_error_world[1])) <= self.stage_a_descent_unlock_y
        )
        hover_target_world = cube_pos.copy()
        hover_target_world[2] = cube_pos[2] + self.stage_a_reset_view_local_z_target
        pregrasp_target_world = cube_pos.copy()
        pregrasp_target_world[2] = cube_pos[2] + self.object_profile["grasp_offset"][2]
        desired_world = pregrasp_target_world if xy_ready else hover_target_world
        pos_world = desired_world - eef_pos
        ori_error = self._quat_axis_error(self.current_target_quat, eef_quat)

        action = np.zeros(self.env.action_dim, dtype=np.float32)
        action[:3] = np.clip(pos_world / 0.015, -0.8, 0.8)
        action[3:6] = np.clip(ori_error / 0.10, -0.3, 0.3)
        action[-1] = -1.0
        return action

    def run_scripted_topdown_descend(self, max_steps: int | None = None):
        max_steps = int(max_steps or self.horizon)
        obs = self.last_raw_obs if self.last_raw_obs is not None else self.env._get_observations(force_update=True)
        obs_vec = self.sync_from_raw_observation(obs, update_handoff_streak=False)
        final_info = {}

        for step in range(1, max_steps + 1):
            action = self._build_scripted_topdown_action(obs)
            obs_vec, _, terminated, truncated, info = self.step(action)
            final_info = info
            obs = self.last_raw_obs
            if bool(info.get("success", False)):
                return obs_vec, True, step, final_info
            if terminated or truncated:
                return obs_vec, False, step, final_info

        return obs_vec, bool(final_info.get("success", False)), max_steps, final_info

    def _solve_ik_topdown_start(self, cube_pos):
        self.reset_helper_env.robots[0].init_qpos = self.default_arm_qpos.copy()
        obs = self.reset_helper_env.reset()
        self._set_gripper_fully_open(self.reset_helper_env)
        self._set_free_joint_pose(self.reset_helper_env, self.reset_helper_cube_joint, cube_pos)
        self.reset_helper_env.sim.forward()
        obs = self.reset_helper_env._get_observations(force_update=True)

        if self.default_controller_orientation is None:
            self.default_controller_orientation = np.array(obs["robot0_eef_quat"], dtype=np.float64)
        target_quat = self.default_controller_orientation.copy()

        success = False
        for _ in range(self.reset_ik_steps):
            frame = self._get_gripper_frame_state(obs, env=self.reset_helper_env)
            local_x_error = frame["local_x"] - self.stage_a_reset_view_local_x_target
            local_y_error = frame["local_y"] - self.stage_a_reset_view_local_y_target
            local_z_error = frame["local_z"] - self.stage_a_reset_view_local_z_target
            eef_quat = np.array(obs["robot0_eef_quat"], dtype=np.float64)
            ori_error = self._quat_axis_error(target_quat, eef_quat)

            if (
                abs(local_x_error) <= self.stage_a_reset_view_local_x_tolerance
                and abs(local_y_error) <= self.stage_a_reset_view_local_y_tolerance
                and abs(local_z_error) <= self.stage_a_reset_view_local_z_tolerance
            ):
                success = True
                break

            pos_world = (
                frame["x_axis"] * local_x_error
                + frame["y_axis"] * local_y_error
                + frame["z_axis"] * local_z_error
            )
            action = np.zeros(self.reset_helper_env.action_dim, dtype=np.float32)
            action[:3] = np.clip(pos_world / 0.02, -0.8, 0.8)
            action[3:6] = np.clip(ori_error / 0.10, -0.5, 0.5)
            action[-1] = -1.0
            obs, _, _, _ = self.reset_helper_env.step(action)

        arm_qpos = np.array(
            self.reset_helper_env.sim.data.qpos[self.reset_helper_env.robots[0]._ref_joint_pos_indexes],
            dtype=np.float64,
        )
        return {
            "success": success,
            "attempt": 1,
            "arm_qpos": arm_qpos,
            "target_pos": np.array(obs["robot0_eef_pos"], dtype=np.float64),
            "target_quat": target_quat,
        }

    def _apply_action_constraints(self, obs, action):
        adjusted_action, table_guard_active, _ = super()._apply_action_constraints(obs, action)
        frame = self._get_gripper_frame_state(obs)
        stage = self._get_stage_state(obs)
        local_x = frame["local_x"] - self.stage_a_local_x_target
        local_y = frame["local_y"] - self.stage_a_local_y_target
        local_z_error = frame["local_z"] - self.stage_a_local_z_target
        descent_unlocked = bool(
            abs(local_x) <= self.stage_a_descent_unlock_x
            and abs(local_y) <= self.stage_a_descent_unlock_y
        )
        forced_descend_active = False
        prealign_descend_blocked = False

        xy_limit = self.stage_a_xy_action_limit
        if self.stage_a_workflow_mode == "topdown_descend":
            xy_limit = min(xy_limit, 0.045)
        adjusted_action[0] = float(np.clip(adjusted_action[0], -xy_limit, xy_limit))
        adjusted_action[1] = float(np.clip(adjusted_action[1], -xy_limit, xy_limit))
        adjusted_action[2] = float(np.clip(adjusted_action[2], -self.stage_a_z_action_limit, self.stage_a_z_action_limit))
        if not descent_unlocked and adjusted_action[2] < -self.stage_a_prealign_downward_limit:
            adjusted_action[2] = -self.stage_a_prealign_downward_limit
            prealign_descend_blocked = True
        elif (
            descent_unlocked
            and local_z_error > self.stage_a_local_z_tolerance
            and not stage["any_contact"]
            and not stage["any_table_contact"]
            and adjusted_action[2] > self.stage_a_forced_descend_action
        ):
            adjusted_action[2] = self.stage_a_forced_descend_action
            forced_descend_active = True
        adjusted_action[3:6] = 0.0
        adjusted_action[-1] = min(float(adjusted_action[-1]), -0.5)
        close_assist_active = False
        return adjusted_action, table_guard_active, close_assist_active, descent_unlocked, prealign_descend_blocked, forced_descend_active

    def _compute_reward(self, obs, action):
        stage = self._get_stage_state(obs)
        frame = self._get_gripper_frame_state(obs)

        grasp_dist = float(np.linalg.norm(stage["grasp_delta"]))
        grasp_xy_dist = stage["grasp_xy_dist"]
        vertical_error = stage["vertical_error"]
        table_clearance = float(frame["jaw_center"][2] - self.table_height)
        gripper_cmd = float(action[-1]) if len(action) > 0 else 0.0
        local_x = frame["local_x"] - self.stage_a_local_x_target
        local_y = frame["local_y"] - self.stage_a_local_y_target
        local_z = frame["local_z"]
        local_z_error = local_z - self.stage_a_local_z_target
        pad_gap = frame["pad_gap"]
        unilateral_contact = bool(stage["any_contact"] and not stage["bilateral_contact"])
        asymmetric_contact_penalty = abs(local_y) if unilateral_contact else 0.0
        clean_cage_ready = bool(
            frame["laterally_caged"]
            and frame["fore_aft_caged"]
            and frame["height_aligned"]
            and frame["open_enough"]
            and not stage["any_contact"]
            and not stage["any_table_contact"]
        )

        prev_grasp_dist = self.prev_grasp_dist if self.prev_grasp_dist is not None else grasp_dist
        prev_grasp_xy_dist = self.prev_grasp_xy_dist if self.prev_grasp_xy_dist is not None else grasp_xy_dist
        prev_vertical_error_abs = (
            self.prev_vertical_error_abs if self.prev_vertical_error_abs is not None else abs(vertical_error)
        )
        visual_reference_error = self._get_visual_reference_error()
        prev_visual_reference_error = (
            self.stage_a_reference_visual_error
            if self.stage_a_reference_visual_error is not None
            else visual_reference_error
        )

        grasp_progress = prev_grasp_dist - grasp_dist
        xy_progress = prev_grasp_xy_dist - grasp_xy_dist
        vertical_progress = prev_vertical_error_abs - abs(vertical_error)
        visual_reference_progress = (
            0.0
            if visual_reference_error is None or prev_visual_reference_error is None
            else prev_visual_reference_error - visual_reference_error
        )
        need_descend = bool(local_z_error > self.stage_a_local_z_tolerance)
        safe_to_descend = bool(
            table_clearance > (self.stage_a_table_clearance_threshold + 0.02)
            and not stage["any_contact"]
            and not stage["any_table_contact"]
        )
        descend_command_active = bool(safe_to_descend and need_descend and action[2] < -0.03)
        hovering_above_target = bool(safe_to_descend and need_descend and action[2] > -0.02)
        near_height_band = bool(stage["xy_aligned"] and abs(local_z_error) < 0.025)
        strongly_above_target = bool(local_z_error > (self.stage_a_local_z_tolerance + 0.02))
        aligned_but_high = bool(stage["xy_aligned"] and stage["gripper_open"] and strongly_above_target)
        descent_unlocked = bool(
            abs(local_x) <= self.stage_a_descent_unlock_x
            and abs(local_y) <= self.stage_a_descent_unlock_y
        )

        geometric_success = bool(
            frame["fore_aft_caged"]
            and frame["laterally_caged"]
            and frame["height_aligned"]
            and not stage["any_table_contact"]
            and frame["open_enough"]
            and table_clearance > self.stage_a_table_clearance_threshold
        )
        topdown_ready = bool(
            abs(local_x) <= self.stage_a_local_x_success_threshold
            and abs(local_y) <= self.stage_a_local_y_success_threshold
            and frame["height_aligned"]
            and frame["open_enough"]
            and not stage["any_table_contact"]
        )
        if self.stage_a_workflow_mode == "topdown_descend":
            geometric_success = topdown_ready
        b_handoff_ready_now = bool(
            geometric_success
            and not stage["any_contact"]
        )
        out_of_workspace = bool(
            abs(local_x) > self.stage_a_max_local_x
            or abs(local_y) > self.stage_a_max_local_y
            or local_z < self.stage_a_min_local_z
            or local_z > self.stage_a_max_local_z
        )

        reward_parts = {
            # Goal-conditioned style shaping: staying far away should remain clearly bad.
            "distance_penalty": -3.0 * grasp_dist,
            "local_x_penalty": -5.0 * abs(local_x),
            "local_y_penalty": -6.5 * abs(local_y),
            "local_z_penalty": -3.5 * abs(local_z_error),
            "xy_progress": 3.0 * xy_progress,
            "vertical_progress": (5.0 if stage["xy_aligned"] else 2.5) * vertical_progress,
            "centered_bonus": 0.40 if frame["centered_between_fingers"] else 0.0,
            "fore_aft_bonus": 0.30 if frame["fore_aft_aligned"] else 0.0,
            "lateral_cage_bonus": 0.45 if frame["laterally_caged"] else 0.0,
            "fore_aft_cage_bonus": 0.45 if frame["fore_aft_caged"] else 0.0,
            "cage_enclosed_bonus": 0.55 if frame["cage_enclosed"] else 0.0,
            "descent_unlock_bonus": 0.25 if descent_unlocked else 0.0,
            "height_band_bonus": 0.25 if frame["height_aligned"] else 0.0,
            "approach_height_band_bonus": 0.20 if near_height_band else 0.0,
            "topdown_xy_hold_bonus": 0.30 if (self.stage_a_workflow_mode == "topdown_descend" and stage["xy_aligned"]) else 0.0,
            "clean_cage_bonus": 0.55 if clean_cage_ready else 0.0,
            "usable_pose_bonus": 0.35 if geometric_success else 0.0,
            "b_handoff_ready_bonus": 0.65 if b_handoff_ready_now else 0.0,
            "visual_reference_error_penalty": (
                -self.stage_a_visual_reference_error_scale * float(visual_reference_error)
                if visual_reference_error is not None
                else 0.0
            ),
            "visual_reference_progress_bonus": (
                self.stage_a_visual_reference_progress_scale * float(visual_reference_progress)
                if visual_reference_error is not None
                else 0.0
            ),
            "visual_reference_match_bonus": (
                self.stage_a_visual_reference_match_bonus
                if visual_reference_error is not None
                and visual_reference_error < self.stage_a_visual_reference_match_threshold
                and stage["gripper_open"]
                and not stage["any_table_contact"]
                else 0.0
            ),
            "open_bonus": 0.12 if stage["gripper_open"] else -0.40,
            "pad_gap_bonus": 0.10 if pad_gap > self.stage_a_open_width_threshold else -0.20,
            "table_clearance_bonus": 0.10 if table_clearance > self.stage_a_table_clearance_threshold else 0.0,
            "success": 0.0,
            "descend_when_high_bonus": 0.40 if descend_command_active else 0.0,
            "aligned_descend_bonus": 0.30 if (descend_command_active and aligned_but_high) else 0.0,
            "hover_above_target_penalty": -0.45 if hovering_above_target else 0.0,
            "aligned_but_high_penalty": -0.25 if aligned_but_high and not descend_command_active else 0.0,
            "premature_descend_penalty": -0.30 if (not descent_unlocked and action[2] < -0.03) else 0.0,
            "table_contact_penalty": -1.20 if stage["any_table_contact"] else 0.0,
            "low_clearance_penalty": -0.35 if table_clearance < 0.03 else 0.0,
            "downward_penalty": -0.40 if table_clearance < 0.05 and action[2] < -0.05 else 0.0,
            "close_penalty": -0.45 if gripper_cmd > -0.1 else 0.0,
            "contact_penalty": -0.35 if stage["any_contact"] else 0.0,
            "unilateral_contact_penalty": -0.65 if unilateral_contact else 0.0,
            "asymmetric_contact_penalty": -8.0 * asymmetric_contact_penalty,
            "grasp_penalty": -0.40 if stage["grasped"] else 0.0,
            "workspace_escape_penalty": -1.00 if out_of_workspace else 0.0,
            "action_penalty": -0.01 * float(np.linalg.norm(action[:3])),
            "time_penalty": -0.005,
        }

        reward = sum(reward_parts.values())
        reward_parts["success_flag"] = float(geometric_success)
        reward_parts["grasp_xy_dist"] = grasp_xy_dist
        reward_parts["vertical_error"] = vertical_error
        reward_parts["grasp_dist"] = grasp_dist
        reward_parts["table_clearance"] = table_clearance
        reward_parts["gripper_width"] = stage["gripper_width"]
        reward_parts["local_x"] = local_x
        reward_parts["local_y"] = local_y
        reward_parts["local_z"] = local_z
        reward_parts["local_z_error"] = local_z_error
        reward_parts["pad_gap"] = pad_gap
        reward_parts["cube_support_x"] = frame["cube_support_x"]
        reward_parts["cube_support_y"] = frame["cube_support_y"]
        reward_parts["lateral_half_gap"] = frame["lateral_half_gap"]
        reward_parts["lateral_cage_limit"] = frame["lateral_cage_limit"]
        reward_parts["object_x_min"] = frame["object_x_min"]
        reward_parts["object_x_max"] = frame["object_x_max"]
        reward_parts["laterally_caged"] = float(frame["laterally_caged"])
        reward_parts["fore_aft_caged"] = float(frame["fore_aft_caged"])
        reward_parts["cage_enclosed"] = float(frame["cage_enclosed"])
        reward_parts["centered_between_fingers"] = float(frame["centered_between_fingers"])
        reward_parts["fore_aft_aligned"] = float(frame["fore_aft_aligned"])
        reward_parts["height_band_aligned"] = float(frame["height_aligned"])
        reward_parts["xy_aligned"] = float(stage["xy_aligned"])
        reward_parts["height_aligned"] = float(stage["height_aligned"])
        reward_parts["in_grasp_zone"] = float(stage["in_grasp_zone"])
        reward_parts["any_table_contact"] = float(stage["any_table_contact"])
        reward_parts["any_contact"] = float(stage["any_contact"])
        reward_parts["unilateral_contact"] = float(unilateral_contact)
        reward_parts["clean_cage_ready"] = float(clean_cage_ready)
        reward_parts["geometric_success"] = float(geometric_success)
        reward_parts["topdown_ready"] = float(topdown_ready)
        reward_parts["b_handoff_ready"] = float(b_handoff_ready_now)
        reward_parts["visual_reference_available"] = float(visual_reference_error is not None)
        reward_parts["visual_reference_error"] = (
            float(visual_reference_error) if visual_reference_error is not None else -1.0
        )
        reward_parts["visual_reference_progress_value"] = float(visual_reference_progress)
        reward_parts["descent_unlocked"] = float(descent_unlocked)
        reward_parts["descend_command_active"] = float(descend_command_active)
        reward_parts["hovering_above_target"] = float(hovering_above_target)
        reward_parts["need_descend"] = float(need_descend)
        reward_parts["aligned_but_high"] = float(aligned_but_high)
        reward_parts["out_of_workspace"] = float(out_of_workspace)
        reward_parts["grasp_progress_value"] = grasp_progress
        reward_parts["xy_progress_value"] = xy_progress
        reward_parts["vertical_progress_value"] = vertical_progress
        reward_parts["lift_amount"] = 0.0
        return reward, reward_parts, b_handoff_ready_now

    def step(self, action):
        current_obs = self.last_raw_obs if self.last_raw_obs is not None else self.env._get_observations(force_update=True)
        (
            adjusted_action,
            table_guard_active,
            close_assist_active,
            descent_unlocked,
            prealign_descend_blocked,
            forced_descend_active,
        ) = self._apply_action_constraints(current_obs, action)
        obs, _, done, info = self.env.step(adjusted_action)
        obs_vec = self.sync_from_raw_observation(obs, update_handoff_streak=True)
        handoff_info = self.last_stage_a_handoff
        reward, reward_parts, b_handoff_ready_now = self._compute_reward(obs, adjusted_action)
        self.prev_grasp_dist = reward_parts["grasp_dist"]
        self.prev_grasp_xy_dist = reward_parts["grasp_xy_dist"]
        self.prev_vertical_error_abs = abs(reward_parts["vertical_error"])
        self.stage_a_reference_visual_error = self._get_visual_reference_error()
        self.prev_lift_amount = 0.0
        self.stage_a_b_handoff_streak = self.stage_a_b_handoff_streak + 1 if b_handoff_ready_now else 0
        geometric_success = bool(reward_parts.get("geometric_success", 0.0) > 0.0)
        success = geometric_success
        if success:
            reward += 4.0
            reward_parts["success"] = 4.0
        reward_parts["success_flag"] = float(success)

        info = dict(info)
        info.update(reward_parts)
        info.update(self.last_reset_meta)
        info["success"] = success
        info["table_guard_active"] = table_guard_active
        info["close_assist_active"] = close_assist_active
        info["raw_gripper_action"] = float(action[-1]) if len(action) > 0 else 0.0
        info["adjusted_gripper_action"] = float(adjusted_action[-1]) if len(adjusted_action) > 0 else 0.0
        info["descent_unlocked"] = bool(descent_unlocked)
        info["prealign_descend_blocked"] = bool(prealign_descend_blocked)
        info["forced_descend_active"] = bool(forced_descend_active)
        info["can_handoff_to_stage_a"] = bool(handoff_info["can_handoff_to_stage_a"])
        info["stage_a_handoff_ready_now"] = bool(handoff_info["ready_now"])
        info["stage_a_handoff_consecutive_hits"] = int(handoff_info["consecutive_hits"])
        info["stage_a_handoff_failure_reasons"] = list(handoff_info["failure_reasons"])
        info["camera_name"] = self.camera_name
        info["visual_feature_dim"] = self.visual_feature_dim
        info["geometric_success"] = bool(geometric_success)
        info["stage_a_b_handoff_ready_now"] = bool(b_handoff_ready_now)
        info["stage_a_b_handoff_streak"] = int(self.stage_a_b_handoff_streak)
        info["stage_a_b_handoff_required_steps"] = int(self.stage_a_b_handoff_stable_steps)
        info["stage_a_workflow_mode"] = self.stage_a_workflow_mode
        info.update(self.latest_camera_summary)

        self.stage_a_step_count += 1
        failure_now = bool(reward_parts["out_of_workspace"] > 0.0 or reward_parts["any_table_contact"] > 0.0)
        self.stage_a_failure_streak = self.stage_a_failure_streak + 1 if failure_now else 0
        allow_early_failure = self.stage_a_step_count >= self.stage_a_min_episode_steps
        failure_terminated = bool(
            allow_early_failure and self.stage_a_failure_streak >= self.stage_a_failure_patience_steps
        )
        info["stage_a_step_count"] = self.stage_a_step_count
        info["stage_a_failure_streak"] = self.stage_a_failure_streak
        info["stage_a_failure_terminated"] = failure_terminated
        terminated = bool(done or success or failure_terminated)
        truncated = False
        return obs_vec, float(reward), terminated, truncated, info

    def reset(self, seed=None, options=None):
        self.stage_a_step_count = 0
        self.stage_a_failure_streak = 0
        self.stage_a_b_handoff_streak = 0
        self.stage_a_reference_visual_error = None
        if self.stage_a_reset_strategy != "teleop_like":
            return super().reset(seed=seed, options=options)

        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.reset_count += 1
        self.stage_a_handoff_gate.reset()

        cube_pos = self._sample_cube_position()
        bank_meta = {
            "bank_size": len(self.stage_a_reset_bank),
            "attempts_used": 0,
            "successes": len(self.stage_a_reset_bank),
            "filter_passes": len(self.stage_a_reset_bank),
            "refreshed": False,
        }
        if self._should_refresh_stage_a_reset_bank(cube_pos):
            bank_meta = self._rebuild_stage_a_reset_bank(cube_pos)
            bank_meta["refreshed"] = True
        self._refresh_stage_a_visual_reference(cube_pos)

        if not self.stage_a_reset_bank:
            return super().reset(seed=seed, options=options)

        self.reset_successes += 1
        sampled_reset = self._sample_stage_a_reset_state()
        obs_vec, reset_meta = self._finalize_reset(
            cube_pos=cube_pos,
            arm_qpos=sampled_reset["arm_qpos"],
            target_pos=sampled_reset["target_pos"],
            target_quat=sampled_reset["target_quat"],
            reset_success=True,
            reset_attempts=max(int(bank_meta["attempts_used"]), 1),
            reset_strategy="teleop_like",
            extra_meta={
                "ik_success": bool(bank_meta["successes"] > 0),
                "ik_attempts": int(bank_meta["attempts_used"]),
                "reset_helper_mode": "teleop_like_bank",
                "teleop_like_bank_size": int(bank_meta["bank_size"]),
                "teleop_like_bank_refreshed": bool(bank_meta["refreshed"]),
                "teleop_like_bank_reuse_count": int(self.stage_a_reset_reuse_count),
                "teleop_like_sample_index": int(sampled_reset["sample_index"]),
                "teleop_like_filter_passes": int(bank_meta["filter_passes"]),
                "stage_a_workflow_mode": self.stage_a_workflow_mode,
            },
        )
        self.stage_a_reference_visual_error = self._get_visual_reference_error()
        reset_meta["visual_reference_available"] = bool(self.stage_a_reference_visual_feature is not None)
        reset_meta["visual_reference_error"] = (
            float(self.stage_a_reference_visual_error)
            if self.stage_a_reference_visual_error is not None
            else None
        )
        return obs_vec, reset_meta
