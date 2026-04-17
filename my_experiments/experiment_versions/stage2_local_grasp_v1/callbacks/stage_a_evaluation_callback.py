"""Evaluation callback for Stage A cage training."""

from __future__ import annotations

import json
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from core.stage_a_cage_env import StageACageEnv


class StageAEvaluationCallback(BaseCallback):
    """Runs deterministic evaluation for Stage A and logs cage-oriented metrics."""

    def __init__(
        self,
        eval_env_kwargs: dict,
        eval_freq: int = 20000,
        n_eval_episodes: int = 10,
        metrics_path: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_kwargs = dict(eval_env_kwargs)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.metrics_path = metrics_path
        self.eval_env = None
        self.next_eval_step = self.eval_freq

    def _init_callback(self) -> None:
        self.eval_env = StageACageEnv(**self.eval_env_kwargs)

    def _on_step(self) -> bool:
        current_steps = int(self.model.num_timesteps)
        if self.eval_freq > 0 and current_steps >= self.next_eval_step:
            metrics = self._run_evaluation()
            self.logger.record("eval/success_rate", metrics["success_rate"])
            self.logger.record("eval/mean_reward", metrics["mean_reward"])
            self.logger.record("eval/mean_episode_steps", metrics["mean_episode_steps"])
            self.logger.record("eval/pregrasp_success_rate", metrics["pregrasp_success_rate"])
            self.logger.record("eval/table_contact_rate", metrics["table_contact_rate"])
            self.logger.record("eval/mean_grasp_xy_dist", metrics["mean_grasp_xy_dist"])
            self.logger.record("eval/mean_vertical_error_abs", metrics["mean_vertical_error_abs"])
            self.logger.record("eval/mean_local_x_abs", metrics["mean_local_x_abs"])
            self.logger.record("eval/mean_local_y_abs", metrics["mean_local_y_abs"])
            self.logger.record("eval/mean_local_z_error_abs", metrics["mean_local_z_error_abs"])
            self.logger.record("eval/centered_between_fingers_rate", metrics["centered_between_fingers_rate"])
            self.logger.record("eval/fore_aft_aligned_rate", metrics["fore_aft_aligned_rate"])
            self.logger.record("eval/laterally_caged_rate", metrics["laterally_caged_rate"])
            self.logger.record("eval/fore_aft_caged_rate", metrics["fore_aft_caged_rate"])
            self.logger.record("eval/cage_enclosed_rate", metrics["cage_enclosed_rate"])
            self.logger.record("eval/height_band_aligned_rate", metrics["height_band_aligned_rate"])
            self.logger.record("eval/unilateral_contact_rate", metrics["unilateral_contact_rate"])
            self.logger.record("eval/clean_cage_ready_rate", metrics["clean_cage_ready_rate"])
            self.logger.record("eval/b_handoff_ready_rate", metrics["b_handoff_ready_rate"])

            if self.metrics_path:
                payload = dict(metrics)
                payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self.metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if self.verbose > 0:
                print(
                    "\n[Stage A Eval] "
                    f"steps={metrics['timesteps']:,}, "
                    f"success_rate={metrics['success_rate']:.2f}, "
                    f"table_contact_rate={metrics['table_contact_rate']:.2f}, "
                    f"centered_rate={metrics['centered_between_fingers_rate']:.2f}, "
                    f"clean_cage_rate={metrics['clean_cage_ready_rate']:.2f}, "
                    f"b_handoff_rate={metrics['b_handoff_ready_rate']:.2f}"
                )

            while self.next_eval_step <= current_steps:
                self.next_eval_step += self.eval_freq
        return True

    def _run_evaluation(self):
        rewards = []
        steps_list = []
        successes = 0
        pregrasp_successes = 0
        table_contact_episodes = 0
        final_xy = []
        final_vertical = []
        final_local_x = []
        final_local_y = []
        final_local_z_error = []
        centered_episodes = 0
        fore_aft_episodes = 0
        laterally_caged_episodes = 0
        fore_aft_caged_episodes = 0
        cage_enclosed_episodes = 0
        height_band_episodes = 0
        unilateral_contact_episodes = 0
        clean_cage_ready_episodes = 0
        b_handoff_ready_episodes = 0

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            pregrasp_successes += int(info.get("pregrasp_success", False))

            done = False
            episode_reward = 0.0
            steps = 0
            saw_table_contact = False
            last_info = {}
            centered = False
            fore_aft = False
            laterally_caged = False
            fore_aft_caged = False
            cage_enclosed = False
            height_band = False
            unilateral_contact = False
            clean_cage_ready = False
            b_handoff_ready = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                last_info = step_info
                saw_table_contact = saw_table_contact or bool(step_info.get("table_contact_penalty", 0.0) < 0.0)
                centered = centered or bool(step_info.get("centered_between_fingers", 0.0) > 0.0)
                fore_aft = fore_aft or bool(step_info.get("fore_aft_aligned", 0.0) > 0.0)
                laterally_caged = laterally_caged or bool(step_info.get("laterally_caged", 0.0) > 0.0)
                fore_aft_caged = fore_aft_caged or bool(step_info.get("fore_aft_caged", 0.0) > 0.0)
                cage_enclosed = cage_enclosed or bool(step_info.get("cage_enclosed", 0.0) > 0.0)
                height_band = height_band or bool(step_info.get("height_band_aligned", 0.0) > 0.0)
                unilateral_contact = unilateral_contact or bool(step_info.get("unilateral_contact", 0.0) > 0.0)
                clean_cage_ready = clean_cage_ready or bool(step_info.get("clean_cage_ready", 0.0) > 0.0)
                b_handoff_ready = b_handoff_ready or bool(step_info.get("b_handoff_ready", 0.0) > 0.0)

            rewards.append(episode_reward)
            steps_list.append(steps)
            successes += int(last_info.get("success", False))
            table_contact_episodes += int(saw_table_contact)
            final_xy.append(float(last_info.get("grasp_xy_dist", 0.0)))
            final_vertical.append(abs(float(last_info.get("vertical_error", 0.0))))
            final_local_x.append(abs(float(last_info.get("local_x", 0.0))))
            final_local_y.append(abs(float(last_info.get("local_y", 0.0))))
            final_local_z_error.append(abs(float(last_info.get("local_z_error", 0.0))))
            centered_episodes += int(centered)
            fore_aft_episodes += int(fore_aft)
            laterally_caged_episodes += int(laterally_caged)
            fore_aft_caged_episodes += int(fore_aft_caged)
            cage_enclosed_episodes += int(cage_enclosed)
            height_band_episodes += int(height_band)
            unilateral_contact_episodes += int(unilateral_contact)
            clean_cage_ready_episodes += int(clean_cage_ready)
            b_handoff_ready_episodes += int(b_handoff_ready)

        return {
            "timesteps": int(self.model.num_timesteps),
            "success_rate": successes / max(self.n_eval_episodes, 1),
            "mean_reward": float(np.mean(rewards)),
            "mean_episode_steps": float(np.mean(steps_list)),
            "pregrasp_success_rate": pregrasp_successes / max(self.n_eval_episodes, 1),
            "table_contact_rate": table_contact_episodes / max(self.n_eval_episodes, 1),
            "mean_grasp_xy_dist": float(np.mean(final_xy)),
            "mean_vertical_error_abs": float(np.mean(final_vertical)),
            "mean_local_x_abs": float(np.mean(final_local_x)),
            "mean_local_y_abs": float(np.mean(final_local_y)),
            "mean_local_z_error_abs": float(np.mean(final_local_z_error)),
            "centered_between_fingers_rate": centered_episodes / max(self.n_eval_episodes, 1),
            "fore_aft_aligned_rate": fore_aft_episodes / max(self.n_eval_episodes, 1),
            "laterally_caged_rate": laterally_caged_episodes / max(self.n_eval_episodes, 1),
            "fore_aft_caged_rate": fore_aft_caged_episodes / max(self.n_eval_episodes, 1),
            "cage_enclosed_rate": cage_enclosed_episodes / max(self.n_eval_episodes, 1),
            "height_band_aligned_rate": height_band_episodes / max(self.n_eval_episodes, 1),
            "unilateral_contact_rate": unilateral_contact_episodes / max(self.n_eval_episodes, 1),
            "clean_cage_ready_rate": clean_cage_ready_episodes / max(self.n_eval_episodes, 1),
            "b_handoff_ready_rate": b_handoff_ready_episodes / max(self.n_eval_episodes, 1),
        }

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
