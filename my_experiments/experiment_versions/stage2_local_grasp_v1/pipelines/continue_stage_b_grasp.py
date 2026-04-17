"""Continue training Stage B grasp PPO."""

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
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
EXPERIMENTS_DIR = os.path.dirname(PROJECT_DIR)
for path in (PROJECT_DIR, EXPERIMENTS_DIR):
    if path not in sys.path:
        sys.path.append(path)

from run_tracking import RunTracker
from callbacks.stage_b_evaluation_callback import StageBEvaluationCallback
from core.stage_b_grasp_env import StageBGraspEnv
from core.vision_interface import resolve_stage2_context

# Keep in sync with train_stage_b_grasp.STAGE_B_PPO_DEFAULTS (stability recipe).
STAGE_B_PPO_DEFAULTS = {
    "learning_rate": 2e-5,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.05,
    "ent_coef": 0.008,
    "vf_coef": 0.5,
    "max_grad_norm": 0.2,
    "n_epochs": 5,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Continue Stage B grasp PPO.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--additional-timesteps", type=int, default=100000)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--object-profile", type=str, default=None)
    parser.add_argument("--object-policy", type=str, default=None, choices=["fixed", "small_random"])
    parser.add_argument("--camera-profile", type=str, default=None)
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--n-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--n-epochs", type=int, default=None)
    parser.add_argument("--eval-freq", type=int, default=10000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser.parse_args()


def build_vec_env(n_envs, tracker, resolved_profile, object_policy, vision_context, seed):
    def make_env(rank):
        def _init():
            env = StageBGraspEnv(
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
        os.path.join(PROJECT_DIR, "training_runs", "completed", "stage_b_grasp_*", "final_model_steps_*.zip"),
        os.path.join(PROJECT_DIR, "training_runs", "interrupted", "stage_b_grasp_*", "interrupted_model_steps_*.zip"),
        os.path.join(PROJECT_DIR, "training_runs", "*", "stage_b_grasp_*", "checkpoints", "*.zip"),
    ]
    candidates = []
    for pattern in patterns:
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError("没有找到可继续训练的 Stage B 模型或检查点。")
    candidates.sort(key=lambda path: (os.path.getmtime(path), checkpoint_priority(path), extract_steps(path)))
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
    n_envs = args.n_envs or base_config.get("n_envs", 1)

    vision_context = resolve_stage2_context(
        camera_profile_name=camera_profile,
        vision_input_path=args.vision_input or base_config.get("vision_input"),
        fallback_object_profile=object_profile,
    )
    resolved_profile = vision_context["object_profile_name"]

    run_timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_name = f"stage_b_grasp_continue_{resolved_profile}_{run_timestamp}"

    tracker = RunTracker(
        experiment_dir=PROJECT_DIR,
        run_name=run_name,
        script_name=os.path.basename(__file__),
        purpose="Stage B 闭爪抓住继续训练",
    )
    tracker.record_base_checkpoint(checkpoint_path)

    print("=" * 72)
    print("Stage B: 闭爪抓住继续训练")
    print("=" * 72)
    print(f"基础模型：{checkpoint_path}")
    print(f"本次训练目录：{tracker.run_dir}")
    print(f"TensorBoard 日志目录：{tracker.tensorboard_dir}")
    print(f"并行环境数：{n_envs}")

    env = build_vec_env(
        n_envs=n_envs,
        tracker=tracker,
        resolved_profile=resolved_profile,
        object_policy=object_policy,
        vision_context=vision_context,
        seed=seed,
    )
    model = PPO.load(checkpoint_path)
    model.set_env(env)
    model.tensorboard_log = tracker.tensorboard_dir

    env_obs_shape = tuple(env.observation_space.shape)
    model_obs_shape = tuple(model.observation_space.shape)
    if env_obs_shape != model_obs_shape:
        raise RuntimeError(
            "所选 Stage B 模型与当前环境定义不兼容。"
            f" 模型 observation shape={model_obs_shape}, 当前环境 shape={env_obs_shape}。"
            " 请改用最新一次 Stage B 训练得到的模型继续训练，或重新训练 Stage B。"
        )

    stored_ppo = base_config.get("ppo_hyperparameters", {})
    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else STAGE_B_PPO_DEFAULTS["learning_rate"]
    )
    n_steps = args.n_steps if args.n_steps is not None else stored_ppo.get("n_steps", 1024)
    batch_size = args.batch_size if args.batch_size is not None else stored_ppo.get("batch_size", 128)
    n_epochs = (
        args.n_epochs
        if args.n_epochs is not None
        else STAGE_B_PPO_DEFAULTS["n_epochs"]
    )

    model.learning_rate = learning_rate
    model._setup_lr_schedule()
    model.n_steps = n_steps
    model.batch_size = batch_size
    model.n_epochs = n_epochs
    model.gamma = STAGE_B_PPO_DEFAULTS["gamma"]
    model.gae_lambda = STAGE_B_PPO_DEFAULTS["gae_lambda"]
    # SB3 expects clip_range to be a FloatSchedule (callable); assigning a bare float breaks train().
    model.clip_range = FloatSchedule(STAGE_B_PPO_DEFAULTS["clip_range"])
    model.ent_coef = STAGE_B_PPO_DEFAULTS["ent_coef"]
    model.vf_coef = STAGE_B_PPO_DEFAULTS["vf_coef"]
    model.max_grad_norm = STAGE_B_PPO_DEFAULTS["max_grad_norm"]

    config = {
        "run_name": run_name,
        "resume_from": checkpoint_path,
        "resume_from_config": base_config_path,
        "object_profile": resolved_profile,
        "object_policy": object_policy,
        "camera_profile": camera_profile,
        "vision_input": vision_context["vision_input_path"],
        "vision_labels": vision_context["vision_labels"],
        "policy_uses_visual": vision_context.get("policy_uses_visual", False),
        "vision_role": vision_context.get("vision_role", "classification_only"),
        "seed": seed,
        "n_envs": n_envs,
        "additional_timesteps": args.additional_timesteps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "algorithm": "PPO",
        "controller": "OSC_POSE",
        "training_stage": "stage_b_grasp_continue",
        "ppo_hyperparameters": {
            **STAGE_B_PPO_DEFAULTS,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
        },
    }

    config_path = tracker.path_for("training_config.json")
    with open(config_path, "w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2, ensure_ascii=False)

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, 50000 // n_envs),
        save_path=tracker.checkpoint_dir,
        name_prefix="stage_b_grasp",
    )
    evaluation_callback = StageBEvaluationCallback(
        eval_env_kwargs={
            "object_profile_name": resolved_profile,
            "object_policy": object_policy,
            "vision_context": vision_context,
            "has_renderer": False,
            "seed": seed,
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
            notes=f"Stage B 继续训练完成，耗时 {time.time() - start_time:.1f} 秒。",
        )
        print(f"\nStage B 继续训练完成，最终目录：{final_dir}")
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
        print(f"\nStage B 继续训练被中断，结果已归档到：{final_dir}")
    except Exception as exc:
        final_steps = model.num_timesteps
        final_dir = tracker.finalize(
            status="failed",
            final_steps=final_steps,
            artifacts={"training_config": config_path},
            error_message=str(exc),
            notes="Stage B 继续训练异常退出，请查看 run_info.json。",
        )
        print(f"\nStage B 继续训练失败，结果已归档到：{final_dir}")
        raise
    finally:
        env.close()


if __name__ == "__main__":
    main()
