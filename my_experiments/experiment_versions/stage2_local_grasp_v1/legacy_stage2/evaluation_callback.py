"""Evaluation callback for stage-2 local grasp training."""

from __future__ import annotations

import json
import os
import time

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from core.stage2_local_grasp_env import Stage2LocalGraspEnv


class Stage2EvaluationCallback(BaseCallback):
    """Runs periodic deterministic evaluations and logs success-oriented metrics."""

    def __init__(
        self,
        eval_env_kwargs: dict,
        eval_freq: int = 20000,
        n_eval_episodes: int = 10,
        render_eval_freq: int = 0,
        render_eval_episodes: int = 1,
        render_step_sleep: float = 0.02,
        render_max_steps: int = 220,
        metrics_path: str | None = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.eval_env_kwargs = dict(eval_env_kwargs)
        self.eval_freq = int(eval_freq)
        self.n_eval_episodes = int(n_eval_episodes)
        self.render_eval_freq = int(render_eval_freq)
        self.render_eval_episodes = int(render_eval_episodes)
        self.render_step_sleep = float(render_step_sleep)
        self.render_max_steps = int(render_max_steps)
        self.metrics_path = metrics_path
        self.eval_env = None
        self.next_eval_step = self.eval_freq
        self.next_render_eval_step = self.render_eval_freq if self.render_eval_freq > 0 else None
        self.render_env = None
        self.render_enabled = False

    def _init_callback(self) -> None:
        self.eval_env = Stage2LocalGraspEnv(**self.eval_env_kwargs)
        self.render_enabled = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if self.render_eval_freq > 0 and self.render_enabled:
            render_env_kwargs = dict(self.eval_env_kwargs)
            render_env_kwargs["has_renderer"] = True
            self.render_env = Stage2LocalGraspEnv(**render_env_kwargs)
        elif self.render_eval_freq > 0 and self.verbose > 0:
            print("\n[Stage2 Eval] 未检测到显示环境，跳过渲染展示。")

    def _on_step(self) -> bool:
        current_steps = int(self.model.num_timesteps)

        if self.eval_freq > 0 and current_steps >= self.next_eval_step:
            metrics = self._run_evaluation()
            self.logger.record("eval/success_rate", metrics["success_rate"])
            self.logger.record("eval/mean_reward", metrics["mean_reward"])
            self.logger.record("eval/mean_lift_from_reset", metrics["mean_lift_from_reset"])
            self.logger.record("eval/max_lift_from_reset", metrics["max_lift_from_reset"])
            self.logger.record("eval/mean_episode_steps", metrics["mean_episode_steps"])
            self.logger.record("eval/pregrasp_success_rate", metrics["pregrasp_success_rate"])
            self.logger.record("eval/early_close_rate", metrics["early_close_rate"])
            self.logger.record("eval/table_contact_rate", metrics["table_contact_rate"])
            self.logger.record("eval/grasp_stage_rate", metrics["grasp_stage_rate"])

            if self.metrics_path:
                payload = dict(metrics)
                payload["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(self.metrics_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if self.verbose > 0:
                print(
                    "\n[Stage2 Eval] "
                    f"steps={metrics['timesteps']:,}, "
                    f"success_rate={metrics['success_rate']:.2f}, "
                    f"grasp_stage_rate={metrics['grasp_stage_rate']:.2f}, "
                    f"early_close_rate={metrics['early_close_rate']:.2f}, "
                    f"mean_lift={metrics['mean_lift_from_reset']:.4f}, "
                    f"mean_reward={metrics['mean_reward']:.2f}"
                )

            while self.next_eval_step <= current_steps:
                self.next_eval_step += self.eval_freq

        if (
            self.render_env is not None
            and self.next_render_eval_step is not None
            and current_steps >= self.next_render_eval_step
        ):
            render_ok = True
            try:
                self._run_render_showcase()
            except Exception as exc:
                render_ok = False
                if self.verbose > 0:
                    print(f"\n[Stage2 Render] 渲染展示失败，已自动关闭后续展示：{exc}")
                if self.render_env is not None:
                    self.render_env.close()
                    self.render_env = None
                self.next_render_eval_step = None
            while render_ok and self.next_render_eval_step <= current_steps:
                self.next_render_eval_step += self.render_eval_freq

        return True

    def _run_evaluation(self):
        rewards = []
        lifts = []
        steps_list = []
        successes = 0
        pregrasp_successes = 0
        early_close_episodes = 0
        table_contact_episodes = 0
        grasp_stage_episodes = 0

        for _ in range(self.n_eval_episodes):
            obs, info = self.eval_env.reset()
            reset_cube_height = float(self.eval_env.reset_cube_height)
            pregrasp_successes += int(info.get("pregrasp_success", False))

            done = False
            episode_reward = 0.0
            steps = 0
            max_cube_height = float(self.eval_env.last_raw_obs["cube_pos"][2])
            saw_early_close = False
            saw_table_contact = False
            reached_grasp_stage = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = self.eval_env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                max_cube_height = max(max_cube_height, float(self.eval_env.last_raw_obs["cube_pos"][2]))
                saw_early_close = saw_early_close or bool(step_info.get("early_close_penalty", 0.0) < 0.0)
                saw_table_contact = saw_table_contact or bool(step_info.get("table_contact_penalty", 0.0) < 0.0)
                reached_grasp_stage = reached_grasp_stage or bool(step_info.get("stage_grasp", 0.0) > 0.0)

            lift_amount = max_cube_height - reset_cube_height
            success = lift_amount > self.eval_env.object_profile["success_height"]

            rewards.append(episode_reward)
            lifts.append(lift_amount)
            steps_list.append(steps)
            successes += int(success)
            early_close_episodes += int(saw_early_close)
            table_contact_episodes += int(saw_table_contact)
            grasp_stage_episodes += int(reached_grasp_stage)

        return {
            "timesteps": int(self.model.num_timesteps),
            "success_rate": successes / max(self.n_eval_episodes, 1),
            "mean_reward": float(np.mean(rewards)),
            "mean_lift_from_reset": float(np.mean(lifts)),
            "max_lift_from_reset": float(np.max(lifts)),
            "mean_episode_steps": float(np.mean(steps_list)),
            "pregrasp_success_rate": pregrasp_successes / max(self.n_eval_episodes, 1),
            "early_close_rate": early_close_episodes / max(self.n_eval_episodes, 1),
            "table_contact_rate": table_contact_episodes / max(self.n_eval_episodes, 1),
            "grasp_stage_rate": grasp_stage_episodes / max(self.n_eval_episodes, 1),
        }

    def _run_render_showcase(self):
        if self.verbose > 0:
            print(
                "\n[Stage2 Render] "
                f"steps={int(self.model.num_timesteps):,}, "
                f"episodes={self.render_eval_episodes}"
            )

        for episode_idx in range(self.render_eval_episodes):
            obs, info = self.render_env.reset()
            reset_cube_height = float(self.render_env.reset_cube_height)
            done = False
            max_cube_height = float(self.render_env.last_raw_obs["cube_pos"][2])
            steps = 0

            if self.verbose > 0:
                print(f"[Stage2 Render] episode={episode_idx + 1}, reset={info}")

            while not done and steps < self.render_max_steps:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, _ = self.render_env.step(action)
                done = terminated or truncated
                steps += 1
                max_cube_height = max(max_cube_height, float(self.render_env.last_raw_obs["cube_pos"][2]))
                self.render_env.render()
                time.sleep(self.render_step_sleep)

            lift_amount = max_cube_height - reset_cube_height
            success = lift_amount > self.render_env.object_profile["success_height"]
            if self.verbose > 0:
                print(
                    "[Stage2 Render] "
                    f"episode={episode_idx + 1}, steps={steps}, "
                    f"lift_from_reset={lift_amount:.4f}, success={success}"
                )

    def _on_training_end(self) -> None:
        if self.eval_env is not None:
            self.eval_env.close()
        if self.render_env is not None:
            self.render_env.close()
