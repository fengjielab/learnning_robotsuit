"""Visualize the scripted Stage A bootstrap that hands off to Stage B."""

from __future__ import annotations

import argparse
import os
import sys
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from core.stage_b_grasp_env import StageBGraspEnv
from core.vision_interface import resolve_stage2_context


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize scripted Stage A bootstrap.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--end-pause", type=float, default=3.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--object-profile", type=str, default="cube_small")
    parser.add_argument("--object-policy", type=str, default="fixed", choices=["fixed", "small_random"])
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--camera-profile", type=str, default="realsense_d435i")
    return parser.parse_args()


def main():
    args = parse_args()
    vision_context = resolve_stage2_context(
        camera_profile_name=args.camera_profile,
        vision_input_path=args.vision_input,
        fallback_object_profile=args.object_profile,
    )

    print("=" * 72)
    print("使用 scripted Stage A bootstrap")
    print("=" * 72)
    print(f"相机配置：{args.camera_profile}")
    print(f"视觉标签：{vision_context['vision_labels']}")
    print(f"物体模板：{vision_context['object_profile_name']}")

    env = StageBGraspEnv(
        object_profile_name=vision_context["object_profile_name"],
        object_policy=args.object_policy,
        vision_context=vision_context,
        has_renderer=not args.no_render,
    )

    try:
        for episode in range(1, args.episodes + 1):
            _, info = env.reset()
            print("\n" + "-" * 72)
            print(f"第 {episode}/{args.episodes} 次 bootstrap")
            print(
                f"start_ok={bool(info.get('stage_b_start_success', False))}, "
                f"steps={int(info.get('stage_b_alignment_steps', 0))}, "
                f"local_x={float(info.get('stage_b_local_x', 0.0)):.4f}, "
                f"local_y={float(info.get('stage_b_local_y', 0.0)):.4f}, "
                f"local_z={float(info.get('stage_b_local_z', 0.0)):.4f}, "
                f"pad_gap={float(info.get('stage_b_pad_gap', 0.0)):.4f}"
            )

            if not args.no_render:
                for _ in range(max(1, int(args.end_pause / max(args.sleep, 1e-3)))):
                    env.render()
                    time.sleep(args.sleep)
    finally:
        env.close()


if __name__ == "__main__":
    main()
