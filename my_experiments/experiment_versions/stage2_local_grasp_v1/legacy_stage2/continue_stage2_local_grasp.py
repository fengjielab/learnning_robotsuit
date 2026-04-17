"""Continue training a stage-2 local grasp PPO policy."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
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
    parser = argparse.ArgumentParser(description="Continue stage-2 local grasp PPO.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--additional-timesteps", type=int, default=200000)
    parser.add_argument("--object-profile", type=str, default=None)
    parser.add_argument("--object-policy", type=str, default=None, choices=["fixed", "small_random"])
    parser.add_argument("--camera-profile", type=str, default=None)
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=20000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--render-eval-freq", type=int, default=0)
    parser.add_argument("--render-eval-episodes", type=int, default=1)
    parser.add_argument("--render-step-sleep", type=float, default=0.02)
    return parser.parse_args()


def extract_steps(path: str):
    matches = re.findall(r"(\d+)", os.path.basename(path))
    return int(matches[-1]) if matches else 0


def checkpoint_priority(path: str):
    basename = os.path.basename(path)
    if basename.startswith("interrupted_model_steps_"):
        return 3
    if basename.startswith("final_model_steps_"):
        return 2
    return 1


def find_latest_checkpoint():
    patterns = [
        os.path.join(PROJECT_DIR, "training_runs", "*", "*", "final_model_steps_*.zip"),
        os.path.join(PROJECT_DIR, "training_runs", "*", "*", "interrupted_model_steps_*.zip"),
        os.path.join(PROJECT_DIR, "training_runs", "*", "*", "checkpoints", "*.zip"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError("没有找到可继续训练的 stage-2 模型或检查点。")
    candidates.sort(
        key=lambda path: (extract_steps(path), checkpoint_priority(path), os.path.getmtime(path))
    )
    return candidates[-1]


def resolve_training_config_path(checkpoint_path: str):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if os.path.basename(checkpoint_dir) == "checkpoints":
        candidate = os.path.join(os.path.dirname(checkpoint_dir), "training_config.json")
    else:
        candidate = os.path.join(checkpoint_dir, "training_config.json")
    return candidate if os.path.exists(candidate) else None


def load_base_config(checkpoint_path: str):
    config_path = resolve_training_config_path(checkpoint_path)
    if not config_path:
        return {}, None
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle), config_path


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint) if args.checkpoint else find_latest_checkpoint()
    base_config, base_config_path = load_base_config(checkpoint_path)

    object_profile = args.object_profile or base_config.get("object_profile") or "cube_small"
    object_policy = args.object_policy or base_config.get("object_policy") or "fixed"
    camera_profile = args.camera_profile or base_config.get("camera_profile") or "realsense_d435i"
    seed = args.seed if args.seed is not None else base_config.get("seed", 7)

    vision_context = resolve_stage2_context(
        camera_profile_name=camera_profile,
        vision_input_path=args.vision_input or base_config.get("vision_input"),
        fallback_object_profile=object_profile,
    )
    resolved_profile = vision_context["object_profile_name"]

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"stage2_local_grasp_continue_{resolved_profile}_{run_timestamp}"

    tracker = RunTracker(
        experiment_dir=PROJECT_DIR,
        run_name=run_name,
        script_name=os.path.basename(__file__),
        purpose="二阶段局部抓取继续训练",
    )
    tracker.record_base_checkpoint(checkpoint_path)

    print("=" * 72)
    print("Stage-2 局部抓取继续训练")
    print("=" * 72)
    print(f"基础模型：{checkpoint_path}")
    print(f"本次训练目录：{tracker.run_dir}")
    print(f"TensorBoard 日志目录：{tracker.tensorboard_dir}")

    def make_env():
        env = Stage2LocalGraspEnv(
            object_profile_name=resolved_profile,
            object_policy=object_policy,
            vision_context=vision_context,
            has_renderer=False,
            seed=seed,
        )
        return Monitor(env, tracker.log_dir)

    env = DummyVecEnv([make_env])
    model = PPO.load(checkpoint_path)
    model.set_env(env)
    model.tensorboard_log = tracker.tensorboard_dir

    learning_rate = args.learning_rate or base_config.get("ppo_hyperparameters", {}).get("learning_rate", 5e-5)
    batch_size = args.batch_size or base_config.get("ppo_hyperparameters", {}).get("batch_size", 128)
    n_epochs = args.n_epochs or base_config.get("ppo_hyperparameters", {}).get("n_epochs", 10)

    model.learning_rate = learning_rate
    model.batch_size = batch_size
    model.n_epochs = n_epochs

    config = {
        "run_name": run_name,
        "resume_from": checkpoint_path,
        "resume_from_config": base_config_path,
        "object_profile": resolved_profile,
        "object_policy": object_policy,
        "camera_profile": camera_profile,
        "vision_input": vision_context["vision_input_path"],
        "vision_labels": vision_context["vision_labels"],
        "seed": seed,
        "additional_timesteps": args.additional_timesteps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "render_eval_freq": args.render_eval_freq,
        "render_eval_episodes": args.render_eval_episodes,
        "render_step_sleep": args.render_step_sleep,
        "algorithm": "PPO",
        "controller": "OSC_POSE",
        "training_stage": "stage2_local_grasp_continue",
        "ppo_hyperparameters": {
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        },
    }

    config_path = tracker.path_for("training_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=tracker.checkpoint_dir,
        name_prefix="stage2_local_grasp",
    )
    evaluation_callback = Stage2EvaluationCallback(
        eval_env_kwargs={
            "object_profile_name": resolved_profile,
            "object_policy": object_policy,
            "vision_context": vision_context,
            "has_renderer": False,
            "seed": seed,
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
            total_timesteps=args.additional_timesteps,
            callback=callback,
            tb_log_name=run_name,
            reset_num_timesteps=False,
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
            notes=f"继续训练完成，耗时 {time.time() - start_time:.1f} 秒。",
        )
        print(f"\n继续训练完成，最终目录：{final_dir}")
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
        print(f"\n继续训练被中断，结果已归档到：{final_dir}")
    except Exception as exc:
        final_steps = model.num_timesteps
        final_dir = tracker.finalize(
            status="failed",
            final_steps=final_steps,
            artifacts={"training_config": config_path},
            error_message=str(exc),
            notes="继续训练异常退出，请查看 run_info.json。",
        )
        print(f"\n继续训练失败，结果已归档到：{final_dir}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
