"""Train Stage C: start from a grasped pose and learn to lift / hold."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.dirname(PROJECT_DIR)
for path in (PROJECT_DIR, EXPERIMENTS_DIR):
    if path not in sys.path:
        sys.path.append(path)

from run_tracking import RunTracker
from callbacks.stage_c_evaluation_callback import StageCEvaluationCallback
from core.stage_c_lift_env import StageCLiftEnv
from core.vision_interface import resolve_stage2_context


def parse_args():
    parser = argparse.ArgumentParser(description="Train Stage C lift PPO.")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--object-profile", type=str, default="cube_small")
    parser.add_argument("--object-policy", type=str, default="fixed", choices=["fixed", "small_random"])
    parser.add_argument("--camera-profile", type=str, default="realsense_d435i")
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser.parse_args()


def build_vec_env(n_envs, tracker, resolved_profile, object_policy, vision_context, seed):
    def make_env(rank):
        def _init():
            env = StageCLiftEnv(
                object_profile_name=resolved_profile,
                object_policy=object_policy,
                vision_context=vision_context,
                has_renderer=False,
                seed=seed + rank,
            )
            log_dir = os.path.join(tracker.log_dir, f"env_{rank}")
            os.makedirs(log_dir, exist_ok=True)
            return Monitor(env, log_dir)

        return _init

    env_fns = [make_env(rank) for rank in range(n_envs)]
    if n_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="fork")


def main():
    args = parse_args()
    vision_context = resolve_stage2_context(
        camera_profile_name=args.camera_profile,
        vision_input_path=args.vision_input,
        fallback_object_profile=args.object_profile,
    )
    resolved_profile = vision_context["object_profile_name"]
    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"stage_c_lift_{resolved_profile}_{run_timestamp}"

    tracker = RunTracker(
        experiment_dir=PROJECT_DIR,
        run_name=run_name,
        script_name=os.path.basename(__file__),
        purpose="Stage C 从已抓住状态出发，学习抬起并保持",
    )

    print("=" * 72)
    print("Stage C: 从已抓住状态出发学习抬起并保持")
    print("=" * 72)
    print(f"本次训练目录：{tracker.run_dir}")
    print(f"TensorBoard 日志目录：{tracker.tensorboard_dir}")
    print(f"并行环境数：{args.n_envs}")

    env = build_vec_env(
        n_envs=args.n_envs,
        tracker=tracker,
        resolved_profile=resolved_profile,
        object_policy=args.object_policy,
        vision_context=vision_context,
        seed=args.seed,
    )

    config = {
        "run_name": run_name,
        "object_profile": resolved_profile,
        "object_policy": args.object_policy,
        "camera_profile": args.camera_profile,
        "vision_input": vision_context["vision_input_path"],
        "vision_labels": vision_context["vision_labels"],
        "grasp_condition": vision_context["grasp_condition"],
        "policy_uses_visual": vision_context.get("policy_uses_visual", False),
        "vision_role": vision_context.get("vision_role", "classification_only"),
        "seed": args.seed,
        "n_envs": args.n_envs,
        "total_timesteps": args.total_timesteps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "algorithm": "PPO",
        "controller": "OSC_POSE",
        "training_stage": "stage_c_lift",
        "notes": "Stage C inherits the same classification-selected object profile and impedance template; PPO learns lift and hold from state only.",
        "ppo_hyperparameters": {
            "learning_rate": 4e-5,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": 0.99,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "ent_coef": 0.002,
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
        learning_rate=4e-5,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.002,
        vf_coef=0.5,
        max_grad_norm=0.3,
        seed=args.seed,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50000 // args.n_envs),
        save_path=tracker.checkpoint_dir,
        name_prefix="stage_c_lift",
    )
    evaluation_callback = StageCEvaluationCallback(
        eval_env_kwargs={
            "object_profile_name": resolved_profile,
            "object_policy": args.object_policy,
            "vision_context": vision_context,
            "has_renderer": False,
            "seed": args.seed,
        },
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
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
            notes=f"Stage C 训练完成，耗时 {time.time() - start_time:.1f} 秒。",
        )
        print(f"\nStage C 训练完成，最终目录：{final_dir}")
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
        print(f"\nStage C 训练被中断，结果已归档到：{final_dir}")
    except Exception as exc:
        final_steps = model.num_timesteps
        final_dir = tracker.finalize(
            status="failed",
            final_steps=final_steps,
            artifacts={"training_config": config_path},
            error_message=str(exc),
            notes="Stage C 训练异常退出，请查看 run_info.json。",
        )
        print(f"\nStage C 训练失败，结果已归档到：{final_dir}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
