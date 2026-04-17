"""Run the staged A -> B -> C grasp pipeline with real simulator state transfer."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
for path in (PROJECT_DIR,):
    if path not in sys.path:
        sys.path.append(path)

from core.sb3_model_loader import load_policy_from_training_run
from core.stage_a_cage_env import StageACageEnv
from core.stage_b_grasp_env import StageBGraspEnv
from core.stage_c_lift_env import StageCLiftEnv
from core.vision_interface import resolve_stage2_context

STAGE_TO_ENV = {
    "A": StageACageEnv,
    "B": StageBGraspEnv,
    "C": StageCLiftEnv,
}

STAGE_TO_PATTERNS = {
    "A": ["stage_a_cage_*", "stage_a_cage_continue_*"],
    "B": [
        "stage_b_grasp_*",
        "stage_b_grasp_continue_*",
    ],
    "C": ["stage_c_lift_*", "stage_c_lift_continue_*"],
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run the staged local grasp state machine.")
    parser.add_argument("--stage-a-model", type=str, default=None)
    parser.add_argument("--stage-b-model", type=str, default=None)
    parser.add_argument("--stage-c-model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--stage-pause", type=float, default=1.0)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--camera-profile", type=str, default=None)
    parser.add_argument("--start-stage", type=str, choices=["A", "B", "C"], default="A")
    parser.add_argument("--learned-a", action="store_true", help="Use the learned Stage A policy instead of scripted Stage A bootstrap.")
    parser.add_argument("--max-b-to-a-fallbacks", type=int, default=1)
    parser.add_argument("--max-c-to-b-fallbacks", type=int, default=1)
    parser.add_argument("--max-a-reset-retries", type=int, default=3)
    return parser.parse_args()


def render_stage_intro(env, render: bool, sleep_s: float, stage_pause_s: float):
    if not render:
        return
    for _ in range(3):
        env.render()
        time.sleep(max(0.0, sleep_s))
    if stage_pause_s > 0.0:
        time.sleep(stage_pause_s)


def extract_steps(path: str):
    matches = re.findall(r"(\d+)", os.path.basename(path))
    return int(matches[-1]) if matches else 0


def find_latest_model(stage_name: str):
    candidates = []
    for run_glob in STAGE_TO_PATTERNS[stage_name]:
        base = os.path.join(PROJECT_DIR, "training_runs", "completed")
        pattern = os.path.join(base, run_glob, "final_model_steps_*.zip")
        candidates.extend(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(f"没有找到可用的 Stage {stage_name} 最终模型。")
    candidates.sort(key=os.path.getmtime)
    return candidates[-1]


def load_training_config(model_path: str):
    run_dir = os.path.dirname(model_path)
    config_path = os.path.join(run_dir, "training_config.json")
    if not os.path.exists(config_path):
        return {}, None
    with open(config_path, "r", encoding="utf-8") as handle:
        return json.load(handle), config_path


def build_common_context(args, configs: list[dict]):
    merged = {}
    for cfg in configs:
        merged.update({k: v for k, v in cfg.items() if v is not None})

    object_profile = merged.get("object_profile", "cube_small")
    object_policy = merged.get("object_policy", "fixed")
    camera_profile = args.camera_profile or merged.get("camera_profile") or "realsense_d435i"
    vision_context = resolve_stage2_context(
        camera_profile_name=camera_profile,
        vision_input_path=args.vision_input or merged.get("vision_input"),
        fallback_object_profile=object_profile,
    )
    return object_profile, object_policy, camera_profile, vision_context


def create_stage_env(stage_name: str, object_profile: str, object_policy: str, vision_context: dict, render: bool):
    env_cls = STAGE_TO_ENV[stage_name]
    return env_cls(
        object_profile_name=object_profile,
        object_policy=object_policy,
        vision_context=vision_context,
        has_renderer=render,
    )


def create_scripted_a_entry(
    object_profile: str,
    object_policy: str,
    vision_context: dict,
    render: bool,
    max_retries: int,
    sleep_s: float = 0.0,
    stage_pause_s: float = 0.0,
):
    for attempt in range(1, max_retries + 1):
        stage_a_env = create_stage_env("A", object_profile, object_policy, vision_context, render)
        try:
            _, reset_info = stage_a_env.reset()
            if render:
                scripted_ok = False
                scripted_steps = 0
                scripted_final_info = {}
                while scripted_steps < stage_a_env.horizon:
                    _, scripted_ok, _, scripted_final_info = stage_a_env.run_scripted_topdown_descend(max_steps=1)
                    scripted_steps += 1
                    stage_a_env.render()
                    time.sleep(sleep_s)
                    if scripted_ok or bool(scripted_final_info.get("terminated", False)) or bool(
                        scripted_final_info.get("truncated", False)
                    ):
                        break
            else:
                _, scripted_ok, scripted_steps, scripted_final_info = stage_a_env.run_scripted_topdown_descend()
            scripted_info = dict(reset_info)
            scripted_info.update(
                {
                    "pipeline_entry_stage": "A_scripted",
                    "scripted_stage_a": True,
                    "scripted_stage_a_attempt": attempt,
                    "scripted_stage_a_steps": scripted_steps,
                    "scripted_stage_a_success": bool(scripted_ok),
                    "scripted_stage_a_final_xy": float(scripted_final_info.get("grasp_xy_dist", 0.0)),
                    "scripted_stage_a_final_z": float(abs(scripted_final_info.get("vertical_error", 0.0))),
                }
            )
            if not scripted_ok:
                stage_a_env.close()
                if attempt == max_retries:
                    return None, None, scripted_info, False
                continue

            flattened = capture_flattened_state(stage_a_env)
        finally:
            if 'stage_a_env' in locals():
                stage_a_env.close()

        env = create_stage_env("B", object_profile, object_policy, vision_context, render)
        obs, transfer_info = restore_stage("B", env, flattened, "A_scripted")
        render_stage_intro(env, render, sleep_s, stage_pause_s)
        transfer_info.update(scripted_info)
        transfer_info["transition"] = "A(scripted)->B"
        return env, obs, transfer_info, True

    return None, None, {"pipeline_entry_stage": "A_scripted", "scripted_stage_a_success": False}, False


def capture_flattened_state(stage_env):
    return np.array(stage_env.env.sim.get_state().flatten(), copy=True)


def _restore_base_state(stage_env, flattened_state):
    stage_env.env.sim.set_state_from_flattened(flattened_state)
    stage_env.env.sim.forward()
    obs = stage_env.env._get_observations(force_update=True)
    stage_env.last_raw_obs = obs
    stage_env.current_target_pos = np.array(obs["robot0_eef_pos"], dtype=np.float64)
    return obs


def restore_stage_a(stage_env, flattened_state, source_stage: str):
    stage_env.reset()
    obs = _restore_base_state(stage_env, flattened_state)
    stage_env.reset_cube_height = float(obs["cube_pos"][2])
    stage_env.current_cube_pos = np.array(obs["cube_pos"], dtype=np.float64)
    grasp_goal, grasp_delta, grasp_xy_dist, vertical_error, _ = stage_env._get_grasp_zone_state(obs)
    stage_env.prev_grasp_dist = float(np.linalg.norm(grasp_delta))
    stage_env.prev_grasp_xy_dist = float(grasp_xy_dist)
    stage_env.prev_vertical_error_abs = float(abs(vertical_error))
    stage_env.prev_lift_amount = 0.0
    obs_vec = stage_env._build_observation(obs)
    stage_env.last_reset_meta = {
        "transferred_from_stage": source_stage,
        "pipeline_entry_stage": "A",
        "pipeline_state_transfer": True,
        "target_pos": grasp_goal.tolist(),
        "object_profile": stage_env.object_profile_name,
        "object_policy": stage_env.object_policy,
        "reset_helper_mode": f"{stage_env.reset_helper_mode}+pipeline_transfer",
    }
    return obs_vec, dict(stage_env.last_reset_meta)


def restore_stage_b(stage_env, flattened_state, source_stage: str):
    stage_env.reset()
    obs = _restore_base_state(stage_env, flattened_state)
    stage, frame, cage_ready = stage_env._stage_b_cage_ready(obs)
    stage_env.reset_cube_height = float(obs["cube_pos"][2])
    local_dist = float(
        np.linalg.norm(
            [
                frame["local_x"] - stage_env.stage_b_local_x_target,
                frame["local_y"] - stage_env.stage_b_local_y_target,
                frame["local_z"] - stage_env.stage_b_local_z_target,
            ]
        )
    )
    stage_env.prev_grasp_dist = local_dist
    stage_env.prev_grasp_xy_dist = float(abs(frame["local_y"] - stage_env.stage_b_local_y_target))
    stage_env.prev_vertical_error_abs = float(abs(frame["local_z"] - stage_env.stage_b_local_z_target))
    stage_env.prev_lift_amount = 0.0
    stage_env.consecutive_grasp_steps = 0
    obs_vec = stage_env._build_observation(obs)
    stage_env.last_reset_meta = {
        "transferred_from_stage": source_stage,
        "pipeline_entry_stage": "B",
        "pipeline_state_transfer": True,
        "stage_b_start_success": bool(cage_ready),
        "stage_b_alignment_steps": 0,
        "stage_b_local_x": frame["local_x"],
        "stage_b_local_y": frame["local_y"],
        "stage_b_local_z": frame["local_z"],
        "stage_b_pad_gap": frame["pad_gap"],
        "object_profile": stage_env.object_profile_name,
        "object_policy": stage_env.object_policy,
        "reset_helper_mode": f"{stage_env.reset_helper_mode}+pipeline_transfer",
    }
    return obs_vec, dict(stage_env.last_reset_meta)


def restore_stage_c(stage_env, flattened_state, source_stage: str):
    stage_env.reset()
    obs = _restore_base_state(stage_env, flattened_state)
    stage = stage_env._get_stage_state(obs)
    frame = stage_env._get_gripper_frame_state(obs)
    stage_env.reset_cube_height = float(obs["cube_pos"][2])
    stage_env.reset_cube_lowest_z, _ = stage_env._cube_lowest_z(
        np.array(obs["cube_pos"], dtype=np.float64),
        np.array(obs["cube_quat"], dtype=np.float64),
    )
    local_dist = float(
        np.linalg.norm(
            [
                frame["local_x"] - stage_env.stage_c_local_x_target,
                frame["local_y"] - stage_env.stage_c_local_y_target,
                frame["local_z"] - stage_env.stage_c_local_z_target,
            ]
        )
    )
    stage_env.prev_grasp_dist = local_dist
    stage_env.prev_grasp_xy_dist = float(abs(frame["local_y"] - stage_env.stage_c_local_y_target))
    stage_env.prev_vertical_error_abs = float(abs(frame["local_z"] - stage_env.stage_c_local_z_target))
    stage_env.prev_lift_amount = 0.0
    stage_env.consecutive_grasp_steps = stage_env.stage_b_grasp_success_steps if stage["grasped"] else 0
    stage_env.consecutive_lift_success_steps = 0
    obs_vec = stage_env._build_observation(obs)
    stage_env.last_reset_meta = {
        "transferred_from_stage": source_stage,
        "pipeline_entry_stage": "C",
        "pipeline_state_transfer": True,
        "stage_c_start_success": bool(stage["grasped"]),
        "stage_c_close_steps": 0,
        "stage_c_local_x": frame["local_x"],
        "stage_c_local_y": frame["local_y"],
        "stage_c_local_z": frame["local_z"],
        "stage_c_pad_gap": frame["pad_gap"],
        "object_profile": stage_env.object_profile_name,
        "object_policy": stage_env.object_policy,
        "reset_helper_mode": f"{stage_env.reset_helper_mode}+pipeline_transfer",
    }
    return obs_vec, dict(stage_env.last_reset_meta)


def restore_stage(stage_name: str, stage_env, flattened_state, source_stage: str):
    if stage_name == "A":
        return restore_stage_a(stage_env, flattened_state, source_stage)
    if stage_name == "B":
        return restore_stage_b(stage_env, flattened_state, source_stage)
    if stage_name == "C":
        return restore_stage_c(stage_env, flattened_state, source_stage)
    raise ValueError(f"Unsupported stage: {stage_name}")


def load_model(stage_name: str, path: str):
    try:
        return load_policy_from_training_run(path)
    except Exception as exc:
        msg = str(exc)
        if "Unexpected observation shape" in msg or "observation space does not match" in msg:
            raise RuntimeError(
                f"Stage {stage_name} 模型与当前环境定义不兼容，请先重新训练 Stage {stage_name}。"
            ) from exc
        raise


def diagnose_stage_failure(stage_name: str, info: dict):
    if not info:
        return f"Stage {stage_name} 没有返回有效诊断信息。"

    if stage_name == "A":
        if bool(info.get("any_table_contact", 0.0) > 0.0):
            return "A 未通过：下降阶段触桌了。"
        if abs(float(info.get("local_y", 0.0))) > 0.012:
            return "A 未通过：夹爪还没有稳定在物块正上方。"
        if abs(float(info.get("local_z_error", 0.0))) > 0.016:
            return "A 未通过：还没有下降到预抓取高度。"
        return "A 未通过：正上方下降完成度还不够，暂时不能安全交给 B。"

    if stage_name == "B":
        if bool(info.get("any_table_contact", 0.0) > 0.0):
            return "B 未通过：闭合阶段出现桌面接触。"
        if not bool(info.get("bilateral_contact", 0.0) > 0.0):
            return "B 未通过：没有形成双侧接触，抓取姿态还不够对称。"
        if not bool(info.get("grasped", 0.0) > 0.0):
            return "B 未通过：已经接触但还没形成稳定抓住。"
        return "B 未通过：夹住趋势存在，但还没达到可稳定交给 C 的抓持状态。"

    if stage_name == "C":
        if not bool(info.get("grasped", 0.0) > 0.0):
            return "C 未通过：进入抬起阶段后丢失抓取。"
        if bool(info.get("any_table_contact", 0.0) > 0.0):
            return "C 未通过：抬起阶段仍有桌面干扰。"
        if float(info.get("lift_amount", 0.0)) <= 0.0:
            return "C 未通过：已经抓住，但没有真正把物体抬离桌面。"
        if bool(info.get("severe_tilt", 0.0) > 0.0):
            return "C 未通过：抬起时姿态不稳，物体倾斜过大。"
        return "C 未通过：抬起了但还没稳定保持到成功条件。"

    return f"Stage {stage_name} 未通过：暂无更具体诊断。"


def run_episode(
    episode_idx: int,
    models: dict,
    object_profile: str,
    object_policy: str,
    vision_context: dict,
    render: bool,
    sleep_s: float,
    stage_pause_s: float,
    start_stage: str,
    use_learned_a: bool,
    max_a_reset_retries: int,
    max_b_to_a_fallbacks: int,
    max_c_to_b_fallbacks: int,
):
    transitions = []
    if start_stage == "A" and not use_learned_a:
        env, obs, reset_info, scripted_ok = create_scripted_a_entry(
            object_profile=object_profile,
            object_policy=object_policy,
            vision_context=vision_context,
            render=render,
            max_retries=max_a_reset_retries,
            sleep_s=sleep_s,
            stage_pause_s=stage_pause_s,
        )
        current_stage = "B"
        transitions.append("A(scripted)->B")
        if not scripted_ok:
            print("\n" + "=" * 72)
            print(f"Pipeline Episode {episode_idx}")
            print("=" * 72)
            print(f"scripted Stage A bootstrap 失败：{reset_info}")
            env.close()
            return False
    else:
        current_stage = start_stage
        env = create_stage_env(current_stage, object_profile, object_policy, vision_context, render)
        obs, reset_info = env.reset()
        render_stage_intro(env, render, sleep_s, stage_pause_s)
    episode_reward = 0.0
    stage_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
    stage_steps = {"A": 0, "B": 0, "C": 0}
    a_reset_retries = 0
    b_to_a_fallbacks = 0
    c_to_b_fallbacks = 0
    final_info = {}
    success = False

    print("\n" + "=" * 72)
    print(f"Pipeline Episode {episode_idx}")
    print("=" * 72)
    print(f"初始进入 Stage {current_stage}，reset 信息：{reset_info}")

    try:
        while True:
            try:
                action, _ = models[current_stage].predict(obs, deterministic=True)
            except ValueError as exc:
                if "Unexpected observation shape" in str(exc):
                    raise RuntimeError(
                        f"Stage {current_stage} 模型与当前环境定义不兼容，请重新训练这个阶段。"
                    ) from exc
                raise

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_reward += float(reward)
            stage_rewards[current_stage] += float(reward)
            stage_steps[current_stage] += 1
            final_info = info

            if render:
                env.render()
                time.sleep(sleep_s)

            if info.get("success", False):
                if current_stage == "C":
                    success = True
                    break

                next_stage = "B" if current_stage == "A" else "C"
                flattened = capture_flattened_state(env)
                env.close()
                prev_stage = current_stage
                current_stage = next_stage
                env = create_stage_env(current_stage, object_profile, object_policy, vision_context, render)
                obs, transfer_info = restore_stage(current_stage, env, flattened, prev_stage)
                render_stage_intro(env, render, sleep_s, stage_pause_s)
                transitions.append(f"{prev_stage}->{current_stage}")
                print(f"阶段切换：{prev_stage} 成功，进入 Stage {current_stage}，transfer 信息：{transfer_info}")
                continue

            if done:
                if current_stage == "A" and a_reset_retries < max_a_reset_retries:
                    a_reset_retries += 1
                    env.close()
                    current_stage = "A"
                    env = create_stage_env(current_stage, object_profile, object_policy, vision_context, render)
                    obs, reset_info = env.reset()
                    render_stage_intro(env, render, sleep_s, stage_pause_s)
                    transitions.append("A->A(reset)")
                    print(f"Stage A 失败，重新 reset 第 {a_reset_retries}/{max_a_reset_retries} 次：{reset_info}")
                    continue

                if current_stage == "C" and c_to_b_fallbacks < max_c_to_b_fallbacks:
                    c_to_b_fallbacks += 1
                    flattened = capture_flattened_state(env)
                    env.close()
                    prev_stage = current_stage
                    current_stage = "B"
                    env = create_stage_env(current_stage, object_profile, object_policy, vision_context, render)
                    obs, transfer_info = restore_stage(current_stage, env, flattened, prev_stage)
                    render_stage_intro(env, render, sleep_s, stage_pause_s)
                    transitions.append(f"{prev_stage}->B(fallback)")
                    print(f"阶段回退：{prev_stage} 失败，回到 Stage B，transfer 信息：{transfer_info}")
                    continue

                if current_stage == "B" and b_to_a_fallbacks < max_b_to_a_fallbacks:
                    b_to_a_fallbacks += 1
                    prev_stage = current_stage
                    if use_learned_a:
                        flattened = capture_flattened_state(env)
                        env.close()
                        current_stage = "A"
                        env = create_stage_env(current_stage, object_profile, object_policy, vision_context, render)
                        obs, transfer_info = restore_stage(current_stage, env, flattened, prev_stage)
                        render_stage_intro(env, render, sleep_s, stage_pause_s)
                        transitions.append(f"{prev_stage}->A(fallback)")
                        print(f"阶段回退：{prev_stage} 失败，回到 Stage A，transfer 信息：{transfer_info}")
                    else:
                        env.close()
                        env, obs, transfer_info, scripted_ok = create_scripted_a_entry(
                            object_profile=object_profile,
                            object_policy=object_policy,
                            vision_context=vision_context,
                            render=render,
                            max_retries=max_a_reset_retries,
                            sleep_s=sleep_s,
                            stage_pause_s=stage_pause_s,
                        )
                        current_stage = "B"
                        transitions.append(f"{prev_stage}->A(scripted)->B")
                        print(f"阶段回退：{prev_stage} 失败，重新执行 scripted Stage A，transfer 信息：{transfer_info}")
                        if not scripted_ok:
                            env.close()
                            break
                    continue
                break
    finally:
        env.close()

    print(
        f"结束：success={success}, total_reward={episode_reward:.2f}, "
        f"transitions={transitions or ['A_only']}, "
        f"stage_steps={stage_steps}, stage_rewards={{{', '.join(f'{k}: {v:.2f}' for k, v in stage_rewards.items())}}}"
    )
    if final_info:
        print(
            f"最终信息：grasped={bool(final_info.get('grasped', 0.0) > 0.0)}, "
            f"table_contact={bool(final_info.get('any_table_contact', 0.0) > 0.0)}, "
            f"lift={float(final_info.get('lift_amount', 0.0)):.4f}, "
            f"upright_cos={float(final_info.get('upright_cos', 1.0)):.4f}"
        )
        if not success:
            print(f"失败诊断：{diagnose_stage_failure(current_stage, final_info)}")
    return success


def main():
    args = parse_args()
    use_learned_a = args.learned_a
    stage_paths = {
        "A": (os.path.abspath(args.stage_a_model) if args.stage_a_model else find_latest_model("A")) if use_learned_a else None,
        "B": os.path.abspath(args.stage_b_model) if args.stage_b_model else find_latest_model("B"),
        "C": os.path.abspath(args.stage_c_model) if args.stage_c_model else find_latest_model("C"),
    }
    stage_keys_for_context = ["B", "C"] if not use_learned_a else ["A", "B", "C"]
    stage_configs = [load_training_config(stage_paths[s])[0] for s in stage_keys_for_context]
    object_profile, object_policy, camera_profile, vision_context = build_common_context(args, stage_configs)

    print("=" * 72)
    print("运行 A -> B -> C 抓取状态机")
    print("=" * 72)
    if use_learned_a:
        print(f"Stage A 模型：{stage_paths['A']}")
    else:
        print("Stage A 入口：scripted bootstrap")
    print(f"Stage B 模型：{stage_paths['B']}")
    print(f"Stage C 模型：{stage_paths['C']}")
    print(f"相机配置：{camera_profile}")
    print(f"视觉标签：{vision_context['vision_labels']}")
    print(f"抓取条件：{vision_context['grasp_condition']}")
    print(f"物体模板：{vision_context['object_profile_name']}")

    models = {stage: load_model(stage, path) for stage, path in stage_paths.items() if path}

    successes = 0
    for episode in range(1, args.episodes + 1):
        successes += int(
            run_episode(
                episode_idx=episode,
                models=models,
                object_profile=object_profile,
                object_policy=object_policy,
                vision_context=vision_context,
                render=not args.no_render,
                sleep_s=args.sleep,
                stage_pause_s=args.stage_pause,
                start_stage=args.start_stage,
                use_learned_a=use_learned_a,
                max_a_reset_retries=args.max_a_reset_retries,
                max_b_to_a_fallbacks=args.max_b_to_a_fallbacks,
                max_c_to_b_fallbacks=args.max_c_to_b_fallbacks,
            )
        )

    print("\n" + "=" * 72)
    print(f"Pipeline 总结：success_rate={successes}/{args.episodes} = {successes / max(args.episodes, 1):.2f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
