"""Run a trained stage-2 local grasp model for quick evaluation or visualization."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
from stable_baselines3 import PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)
PROJECT_DIR = PARENT_DIR

from core.stage2_local_grasp_env import Stage2LocalGraspEnv
from core.vision_interface import resolve_stage2_context


def parse_args():
    parser = argparse.ArgumentParser(description="Use a trained stage-2 local grasp model.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--camera-profile", type=str, default=None)
    return parser.parse_args()


def extract_steps(path: str):
    matches = re.findall(r"(\d+)", os.path.basename(path))
    return int(matches[-1]) if matches else 0


def find_latest_model():
    pattern = os.path.join(PROJECT_DIR, "training_runs", "completed", "*", "final_model_steps_*.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError("没有找到可用的 stage-2 最终模型。")
    candidates.sort(key=lambda path: (extract_steps(path), os.path.getmtime(path)))
    return candidates[-1]


def load_training_config(model_path: str):
    run_dir = os.path.dirname(model_path)
    config_path = os.path.join(run_dir, "training_config.json")
    if not os.path.exists(config_path):
        return {}, None
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle), config_path


def main():
    args = parse_args()
    model_path = os.path.abspath(args.model) if args.model else find_latest_model()
    training_config, config_path = load_training_config(model_path)

    object_profile = training_config.get("object_profile", "cube_small")
    object_policy = training_config.get("object_policy", "fixed")
    camera_profile = args.camera_profile or training_config.get("camera_profile") or "realsense_d435i"
    vision_context = resolve_stage2_context(
        camera_profile_name=camera_profile,
        vision_input_path=args.vision_input or training_config.get("vision_input"),
        fallback_object_profile=object_profile,
    )

    print("=" * 72)
    print("使用 Stage-2 局部抓取模型")
    print("=" * 72)
    print(f"模型路径：{model_path}")
    if config_path:
        print(f"训练配置：{config_path}")
    print(f"相机配置：{camera_profile}")
    print(f"视觉标签：{vision_context['vision_labels']}")
    print(f"物体模板：{vision_context['object_profile_name']}")

    model = PPO.load(model_path)
    env = Stage2LocalGraspEnv(
        object_profile_name=vision_context["object_profile_name"],
        object_policy=object_policy,
        vision_context=vision_context,
        has_renderer=not args.no_render,
    )

    success_count = 0
    reward_totals = []

    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            episode_reward = 0.0
            max_cube_height = float(env.last_raw_obs["cube_pos"][2])
            reset_cube_height = float(env.reset_cube_height)
            done = False
            steps = 0

            print("\n" + "-" * 72)
            print(f"第 {episode}/{args.episodes} 次测试")
            print(f"reset 信息：{info}")

            while not done:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                episode_reward += reward
                steps += 1
                max_cube_height = max(max_cube_height, float(env.last_raw_obs["cube_pos"][2]))

                if not args.no_render:
                    env.render()
                    time.sleep(args.sleep)

            relative_height = max_cube_height - env.table_height
            lift_amount = max_cube_height - reset_cube_height
            success = lift_amount > env.object_profile["success_height"]
            success_count += int(success)
            reward_totals.append(episode_reward)

            print(
                f"结束：steps={steps}, total_reward={episode_reward:.2f}, "
                f"max_relative_height={relative_height:.4f}, lift_from_reset={lift_amount:.4f}, success={success}"
            )

        print("\n" + "=" * 72)
        print("测试统计")
        print("=" * 72)
        print(f"测试次数：{args.episodes}")
        print(f"成功次数：{success_count}")
        print(f"成功率：{success_count / max(args.episodes, 1) * 100:.1f}%")
        print(f"平均奖励：{np.mean(reward_totals):.2f}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
