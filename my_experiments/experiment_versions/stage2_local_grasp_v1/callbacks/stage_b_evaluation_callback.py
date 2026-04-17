"""Evaluation callback for Stage B grasp training."""

from __future__ import annotations

import json
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from core.stage_b_grasp_env import StageBGraspEnv


class StageBEvaluationCallback(BaseCallback):
    """Runs deterministic evaluation for Stage B and logs grasp-oriented metrics."""

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
        self.eval_env = StageBGraspEnv(**self.eval_env_kwargs)

    def _on_step(self) -> bool:
        current_steps = int(self.model.num_timesteps)
        if self.eval_freq > 0 and current_steps >= self.next_eval_step:
            metrics = self._run_evaluation()
            self.logger.record("eval/success_rate", metrics["success_rate"])
            self.logger.record("eval/mean_reward", metrics["mean_reward"])
            self.logger.record("eval/mean_episode_steps", metrics["mean_episode_steps"])
            self.logger.record("eval/pregrasp_success_rate", metrics["pregrasp_success_rate"])
            self.logger.record("eval/table_contact_rate", metrics["table_contact_rate"])
            self.logger.record("eval/bilateral_contact_rate", metrics["bilateral_contact_rate"])
            self.logger.record("eval/grasped_rate", metrics["grasped_rate"])
            self.logger.record("eval/mean_local_x_abs", metrics["mean_local_x_abs"])
            self.logger.record("eval/mean_local_y_abs", metrics["mean_local_y_abs"])
            self.logger.record("eval/mean_local_z_error_abs", metrics["mean_local_z_error_abs"])
            self.logger.record("eval/mean_gripper_width", metrics["mean_gripper_width"])

            if self.metrics_path:
                payload = dict(metrics)
                payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self.metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if self.verbose > 0:
                print(
                    "\n[Stage B Eval] "
                    f"steps={metrics['timesteps']:,}, "
                    f"success_rate={metrics['success_rate']:.2f}, "
                    f"grasped_rate={metrics['grasped_rate']:.2f}, "
                    f"bilateral_rate={metrics['bilateral_contact_rate']:.2f}, "
                    f"mean_local_y={metrics['mean_local_y_abs']:.4f}, "
                    f"mean_width={metrics['mean_gripper_width']:.4f}"
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
        bilateral_contact_episodes = 0
        grasped_episodes = 0
        final_local_x = []
        final_local_y = []
        final_local_z_error = []
        final_gripper_width = []

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            pregrasp_successes += int(info.get("stage_b_start_success", False))

            done = False
            episode_reward = 0.0
            steps = 0
            saw_table_contact = False
            saw_bilateral = False
            saw_grasp = False
            last_info = {}

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                last_info = step_info
                saw_table_contact = saw_table_contact or bool(step_info.get("any_table_contact", 0.0) > 0.0)
                saw_bilateral = saw_bilateral or bool(step_info.get("bilateral_contact", 0.0) > 0.0)
                saw_grasp = saw_grasp or bool(step_info.get("grasped", 0.0) > 0.0)

            rewards.append(episode_reward)
            steps_list.append(steps)
            successes += int(last_info.get("success", False))
            table_contact_episodes += int(saw_table_contact)
            bilateral_contact_episodes += int(saw_bilateral)
            grasped_episodes += int(saw_grasp)
            final_local_x.append(abs(float(last_info.get("local_x", 0.0))))
            final_local_y.append(abs(float(last_info.get("local_y", 0.0))))
            final_local_z_error.append(abs(float(last_info.get("local_z_error", 0.0))))
            final_gripper_width.append(float(last_info.get("gripper_width", 0.0)))

        return {
            "timesteps": int(self.model.num_timesteps),
            "success_rate": successes / max(self.n_eval_episodes, 1),
            "mean_reward": float(np.mean(rewards)),
            "mean_episode_steps": float(np.mean(steps_list)),
            "pregrasp_success_rate": pregrasp_successes / max(self.n_eval_episodes, 1),
            "table_contact_rate": table_contact_episodes / max(self.n_eval_episodes, 1),
            "bilateral_contact_rate": bilateral_contact_episodes / max(self.n_eval_episodes, 1),
            "grasped_rate": grasped_episodes / max(self.n_eval_episodes, 1),
            "mean_local_x_abs": float(np.mean(final_local_x)),
            "mean_local_y_abs": float(np.mean(final_local_y)),
            "mean_local_z_error_abs": float(np.mean(final_local_z_error)),
            "mean_gripper_width": float(np.mean(final_gripper_width)),
        }

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
