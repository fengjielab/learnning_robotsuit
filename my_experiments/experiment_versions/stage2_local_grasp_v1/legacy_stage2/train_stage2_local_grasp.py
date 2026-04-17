"""Train a stage-2 local grasp PPO policy for Panda on robosuite Lift."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.dirname(PARENT_DIR)
for path in (PARENT_DIR, EXPERIMENTS_DIR):
    if path not in sys.path:
        sys.path.append(path)
PROJECT_DIR = PARENT_DIR

from run_tracking import RunTracker
from evaluation_callback import Stage2EvaluationCallback
from core.stage2_local_grasp_env import Stage2LocalGraspEnv
from core.vision_interface import resolve_stage2_context


def parse_args():
    parser = argparse.ArgumentParser(description="Train stage-2 local grasp PPO.")
    parser.add_argument("--total-timesteps", type=int, default=300000)
    parser.add_argument("--object-profile", type=str, default="cube_small")
    parser.add_argument("--object-policy", type=str, default="fixed", choices=["fixed", "small_random"])
    parser.add_argument("--camera-profile", type=str, default="realsense_d435i")
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--render-eval-freq", type=int, default=0)
    parser.add_argument("--render-eval-episodes", type=int, default=1)
    parser.add_argument("--render-step-sleep", type=float, default=0.02)
    return parser.parse_args()


def main():
    args = parse_args()
    vision_context = resolve_stage2_context(
        camera_profile_name=args.camera_profile,
        vision_input_path=args.vision_input,
        fallback_object_profile=args.object_profile,
    )
    resolved_profile = vision_context["object_profile_name"]
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"stage2_local_grasp_{resolved_profile}_{run_timestamp}"

    tracker = RunTracker(
        experiment_dir=PROJECT_DIR,
        run_name=run_name,
        script_name=os.path.basename(__file__),
        purpose="二阶段局部抓取训练",
    )

    print("=" * 72)
    print("Stage-2 局部抓取训练")
    print("=" * 72)
    print(f"本次训练目录：{tracker.run_dir}")
    print(f"TensorBoard 日志目录：{tracker.tensorboard_dir}")

    def make_env():
        env = Stage2LocalGraspEnv(
            object_profile_name=resolved_profile,
            object_policy=args.object_policy,
            vision_context=vision_context,
            has_renderer=False,
            seed=args.seed,
        )
        return Monitor(env, tracker.log_dir)

    env = DummyVecEnv([make_env])

    config = {
        "run_name": run_name,
        "object_profile": resolved_profile,
        "object_policy": args.object_policy,
        "camera_profile": args.camera_profile,
        "vision_input": vision_context["vision_input_path"],
        "vision_labels": vision_context["vision_labels"],
        "seed": args.seed,
        "total_timesteps": args.total_timesteps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "render_eval_freq": args.render_eval_freq,
        "render_eval_episodes": args.render_eval_episodes,
        "render_step_sleep": args.render_step_sleep,
        "algorithm": "PPO",
        "controller": "OSC_POSE",
        "training_stage": "stage2_local_grasp",
        "notes": "Stage-1 assumed by helper-based local pregrasp reset. RL controls local pose + gripper only.",
        "ppo_hyperparameters": {
            "learning_rate": 5e-5,
            "n_steps": 1024,
            "batch_size": 128,
            "n_epochs": 10,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.3,
        },
    }

    config_path = tracker.path_for("training_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=tracker.tensorboard_dir,
        learning_rate=5e-5,
        n_steps=1024,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.3,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=tracker.checkpoint_dir,
        name_prefix="stage2_local_grasp",
    )
    evaluation_callback = Stage2EvaluationCallback(
        eval_env_kwargs={
            "object_profile_name": resolved_profile,
            "object_policy": args.object_policy,
            "vision_context": vision_context,
            "has_renderer": False,
            "seed": args.seed,
        },
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        render_eval_freq=args.render_eval_freq,
        render_eval_episodes=args.render_eval_episodes,
        render_step_sleep=args.render_step_sleep,
        render_max_steps=220,
        metrics_path=tracker.path_for("evaluation_history.jsonl"),
        verbose=1,
    )
    callback = CallbackList([checkpoint_callback, evaluation_callback])

    start_time = time.time()
    final_steps = None

    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=callback,
            tb_log_name=run_name,
        )
        final_steps = model.num_timesteps
        model_path = tracker.path_for(f"final_model_steps_{final_steps}.zip")
        model.save(model_path)
        final_dir = tracker.finalize(
            status="completed",
            final_steps=final_steps,
            artifacts={
                "final_model": model_path,
                "training_config": config_path,
            },
            notes=f"训练完成，耗时 {time.time() - start_time:.1f} 秒。",
        )
        print(f"\n训练完成，最终目录：{final_dir}")
    except KeyboardInterrupt:
        final_steps = model.num_timesteps
        interrupted_model = tracker.path_for(f"interrupted_model_steps_{final_steps}.zip")
        model.save(interrupted_model)
        final_dir = tracker.finalize(
            status="interrupted",
            final_steps=final_steps,
            artifacts={
                "interrupted_model": interrupted_model,
                "training_config": config_path,
            },
            notes="用户手动中断，保留用于继续训练。",
        )
        print(f"\n训练被中断，结果已归档到：{final_dir}")
    except Exception as exc:
        final_steps = model.num_timesteps
        final_dir = tracker.finalize(
            status="failed",
            final_steps=final_steps,
            artifacts={"training_config": config_path},
            error_message=str(exc),
            notes="训练异常退出，请查看 run_info.json。",
        )
        print(f"\n训练失败，结果已归档到：{final_dir}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
