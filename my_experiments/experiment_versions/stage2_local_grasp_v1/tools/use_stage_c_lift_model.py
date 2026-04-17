"""Run a trained Stage C lift model for visualization or quick evaluation."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

from stable_baselines3 import PPO

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from core.stage_c_lift_env import StageCLiftEnv
from core.vision_interface import resolve_stage2_context


def parse_args():
    parser = argparse.ArgumentParser(description="Use a trained Stage C lift model.")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--camera-profile", type=str, default=None)
    return parser.parse_args()


def extract_steps(path: str):
    matches = re.findall(r"(\d+)", os.path.basename(path))
    return int(matches[-1]) if matches else 0


def find_latest_model():
    pattern = os.path.join(PROJECT_DIR, "training_runs", "completed", "stage_c_lift_*", "final_model_steps_*.zip")
    candidates = glob.glob(pattern)
    if not candidates:
        raise FileNotFoundError("没有找到可用的 Stage C 最终模型。")
    candidates.sort(key=os.path.getmtime)
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
    print("使用 Stage C 抬起并保持模型")
    print("=" * 72)
    print(f"模型路径：{model_path}")
    if config_path:
        print(f"训练配置：{config_path}")
    print(f"相机配置：{camera_profile}")
    print(f"视觉标签：{vision_context['vision_labels']}")
    print(f"物体模板：{vision_context['object_profile_name']}")

    model = PPO.load(model_path)
    env = StageCLiftEnv(
        object_profile_name=vision_context["object_profile_name"],
        object_policy=object_policy,
        vision_context=vision_context,
        has_renderer=not args.no_render,
    )

    try:
        for episode in range(1, args.episodes + 1):
            obs, info = env.reset()
            done = False
            total_reward = 0.0
            steps = 0
            saw_grasp = False
            max_lift = 0.0
            final_info = {}

            print("\n" + "-" * 72)
            print(f"第 {episode}/{args.episodes} 次测试")
            print(f"reset 信息：{info}")

            while not done:
                try:
                    action, _ = model.predict(obs, deterministic=True)
                except ValueError as exc:
                    if "Unexpected observation shape" in str(exc):
                        raise RuntimeError(
                            "当前 Stage C 模型与最新环境定义不兼容。"
                            " 这通常是因为 Stage C observation 定义已经变化，请重新训练 Stage C 后再试用。"
                        ) from exc
                    raise
                obs, reward, terminated, truncated, step_info = env.step(action)
                done = terminated or truncated
                total_reward += reward
                steps += 1
                saw_grasp = saw_grasp or bool(step_info.get("grasped", 0.0) > 0.0)
                max_lift = max(max_lift, float(step_info.get("lift_amount", 0.0)))
                final_info = step_info

                if not args.no_render:
                    env.render()
                    time.sleep(args.sleep)

            print(
                f"结束：steps={steps}, total_reward={total_reward:.2f}, "
                f"success={bool(final_info.get('success', False))}, "
                f"start_ok={bool(info.get('stage_c_start_success', False))}, "
                f"grasped={saw_grasp}, "
                f"table_contact={bool(final_info.get('any_table_contact', 0.0) > 0.0)}, "
                f"final_lift={float(final_info.get('lift_amount', 0.0)):.4f}, "
                f"max_lift={max_lift:.4f}, "
                f"final_local_y={float(final_info.get('local_y', 0.0)):.4f}, "
                f"final_width={float(final_info.get('gripper_width', 0.0)):.4f}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
