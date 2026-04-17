"""Validate the stage-2 reset pipeline by sampling many local-grasp starts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from core.stage2_local_grasp_env import Stage2LocalGraspEnv
from core.vision_interface import resolve_stage2_context


def parse_args():
    parser = argparse.ArgumentParser(description="Validate stage-2 local grasp resets.")
    parser.add_argument("--num-resets", type=int, default=100)
    parser.add_argument("--object-profile", type=str, default="cube_small")
    parser.add_argument("--object-policy", type=str, default="fixed", choices=["fixed", "small_random"])
    parser.add_argument("--camera-profile", type=str, default="realsense_d435i")
    parser.add_argument("--vision-input", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    vision_context = resolve_stage2_context(
        camera_profile_name=args.camera_profile,
        vision_input_path=args.vision_input,
        fallback_object_profile=args.object_profile,
    )
    env = Stage2LocalGraspEnv(
        object_profile_name=vision_context["object_profile_name"],
        object_policy=args.object_policy,
        vision_context=vision_context,
        has_renderer=False,
    )

    distances = []
    target_offsets = []
    ik_success = 0

    try:
        for _ in range(args.num_resets):
            obs, info = env.reset()
            raw_obs = env.last_raw_obs
            eef_pos = np.array(raw_obs["robot0_eef_pos"], dtype=np.float64)
            cube_pos = np.array(raw_obs["cube_pos"], dtype=np.float64)
            distances.append(float(np.linalg.norm(eef_pos - cube_pos)))
            target_offsets.append(info["target_pos"])
            ik_success += int(info["ik_success"])

        summary = {
            "num_resets": args.num_resets,
            "ik_success_rate": ik_success / max(args.num_resets, 1),
            "pregrasp_success_rate": ik_success / max(args.num_resets, 1),
            "mean_eef_cube_distance": float(np.mean(distances)),
            "min_eef_cube_distance": float(np.min(distances)),
            "max_eef_cube_distance": float(np.max(distances)),
            "last_target_pos": target_offsets[-1] if target_offsets else None,
            "object_profile": vision_context["object_profile_name"],
            "object_policy": args.object_policy,
            "reset_helper_mode": env.reset_helper_mode,
            "camera_profile": args.camera_profile,
            "vision_labels": vision_context["vision_labels"],
        }
        print(json.dumps(summary, indent=2, ensure_ascii=False))
    finally:
        env.close()


if __name__ == "__main__":
    main()
