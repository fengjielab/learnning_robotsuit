"""Teleoperate near the object, then auto-handoff into the full A -> B -> C pipeline."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from pynput.keyboard import Listener

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(PROJECT_DIR)))
for path in (PROJECT_DIR, REPO_DIR):
    if path not in sys.path:
        sys.path.append(path)

from core.stage_a_cage_env import StageACageEnv
from pipelines.run_staged_grasp_pipeline import (
    build_common_context,
    capture_flattened_state,
    create_stage_env,
    create_scripted_a_entry,
    diagnose_stage_failure,
    find_latest_model,
    load_model,
    load_training_config,
    render_stage_intro,
    restore_stage,
)
from robosuite.utils.input_utils import input2action


def parse_args():
    parser = argparse.ArgumentParser(description="Teleoperate near the object, then run the full staged grasp pipeline.")
    parser.add_argument("--stage-a-model", type=str, default=None)
    parser.add_argument("--stage-b-model", type=str, default=None)
    parser.add_argument("--stage-c-model", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sleep", type=float, default=0.02)
    parser.add_argument("--stage-pause", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="keyboard", choices=["keyboard", "spacemouse"])
    parser.add_argument("--pos-sensitivity", type=float, default=1.0)
    parser.add_argument("--rot-sensitivity", type=float, default=1.0)
    parser.add_argument("--vision-input", type=str, default=None)
    parser.add_argument("--camera-profile", type=str, default=None)
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--max-b-to-a-fallbacks", type=int, default=1)
    parser.add_argument("--max-c-to-b-fallbacks", type=int, default=1)
    parser.add_argument("--learned-a", action="store_true", help="Use a learned Stage A model instead of the scripted top-down descend stage.")
    return parser.parse_args()


class HandoffOverride:
    """Keeps manual override as a backup, but not the primary path."""

    def __init__(self):
        self.force_handoff = False
        self.listener = Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        try:
            if key.char == "p":
                self.force_handoff = True
        except AttributeError:
            pass

    def consume(self):
        if self.force_handoff:
            self.force_handoff = False
            return True
        return False


def make_device(args, viewer):
    if args.device == "keyboard":
        from robosuite.devices import Keyboard

        device = Keyboard(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)
        if viewer is not None and hasattr(viewer, "add_keypress_callback"):
            viewer.add_keypress_callback(device.on_press)
        return device

    from robosuite.devices import SpaceMouse

    return SpaceMouse(pos_sensitivity=args.pos_sensitivity, rot_sensitivity=args.rot_sensitivity)


def _load_stage_paths(args):
    return {
        "A": (os.path.abspath(args.stage_a_model) if args.stage_a_model else find_latest_model("A")) if args.learned_a else None,
        "B": os.path.abspath(args.stage_b_model) if args.stage_b_model else find_latest_model("B"),
        "C": os.path.abspath(args.stage_c_model) if args.stage_c_model else find_latest_model("C"),
    }


def main():
    args = parse_args()
    stage_paths = _load_stage_paths(args)
    stage_keys_for_context = ("A", "B", "C") if args.learned_a else ("B", "C")
    stage_configs = [load_training_config(stage_paths[stage])[0] for stage in stage_keys_for_context]
    object_profile, object_policy, camera_profile, vision_context = build_common_context(args, stage_configs)
    models = {stage: load_model(stage, path) for stage, path in stage_paths.items() if path}

    print("=" * 72)
    print("遥操作 -> A -> B -> C 分层抓取流水线")
    print("=" * 72)
    if args.learned_a:
        print(f"Stage A 模型：{stage_paths['A']}")
    else:
        print("Stage A：scripted top-down descend")
    print(f"Stage B 模型：{stage_paths['B']}")
    print(f"Stage C 模型：{stage_paths['C']}")
    print(f"相机配置：{camera_profile}")
    print(f"视觉标签：{vision_context['vision_labels']}")
    print(f"抓取条件：{vision_context['grasp_condition']}")
    print("默认流程：人工先送近，满足 A handoff 条件后自动接管。")
    print("备用强制接管键：p")

    override = HandoffOverride()

    for episode in range(1, args.episodes + 1):
        env = StageACageEnv(
            object_profile_name=vision_context["object_profile_name"],
            object_policy=object_policy,
            vision_context=vision_context,
            has_renderer=not args.no_render,
            reset_strategy="teleop_like",
        )
        device = make_device(args, env.env.viewer if hasattr(env.env, "viewer") else None)

        current_stage = "A"
        policy_obs = None
        total_reward = 0.0
        stage_rewards = {"A": 0.0, "B": 0.0, "C": 0.0}
        stage_steps = {"A": 0, "B": 0, "C": 0}
        transitions = ["teleop->A"]
        manual_steps = 0
        final_info = {}
        success = False
        b_to_a_fallbacks = 0
        c_to_b_fallbacks = 0

        try:
            raw_obs, reset_info = env.reset_to_manual_start()
            print("\n" + "=" * 72)
            print(f"Pipeline Episode {episode}")
            print("=" * 72)
            print(f"初始 reset 信息：{reset_info}")

            if not args.no_render:
                env.env.render()
            device.start_control()

            while policy_obs is None:
                action, _ = input2action(device=device, robot=env.env.robots[0], active_arm="right")
                if action is None:
                    print("收到重置信号，本回合结束。")
                    break

                raw_obs, _, done, _ = env.env.step(action)
                env.last_raw_obs = raw_obs
                handoff = env.evaluate_stage_a_handoff(raw_obs, update_streak=True)
                manual_steps += 1

                if not args.no_render:
                    env.env.render()
                    time.sleep(args.sleep)

                if override.consume():
                    if args.learned_a:
                        print("检测到手动强制接管，切换到 Stage A 策略。")
                        policy_obs = env.prime_policy_from_raw_observation(raw_obs, update_handoff_streak=False)
                    else:
                        print("检测到手动强制接管，执行 scripted Stage A 正上方下降。")
                        flattened = capture_flattened_state(env)
                        env.close()
                        env, policy_obs, transfer_info, scripted_ok = create_scripted_a_entry(
                            object_profile=object_profile,
                            object_policy=object_policy,
                            vision_context=vision_context,
                            render=not args.no_render,
                            max_retries=1,
                            sleep_s=args.sleep,
                            stage_pause_s=args.stage_pause,
                        )
                        current_stage = "B"
                        transitions.append("teleop->A(scripted)->B")
                        print(f"scripted Stage A 信息：{transfer_info}")
                        if not scripted_ok:
                            policy_obs = None
                            break
                elif handoff["can_handoff_to_stage_a"]:
                    print(
                        "自动接管成功："
                        f" hits={handoff['consecutive_hits']},"
                        f" reasons={handoff['failure_reasons']}"
                    )
                    if args.learned_a:
                        policy_obs = env.prime_policy_from_raw_observation(raw_obs, update_handoff_streak=False)
                    else:
                        env.close()
                        env, policy_obs, transfer_info, scripted_ok = create_scripted_a_entry(
                            object_profile=object_profile,
                            object_policy=object_policy,
                            vision_context=vision_context,
                            render=not args.no_render,
                            max_retries=1,
                            sleep_s=args.sleep,
                            stage_pause_s=args.stage_pause,
                        )
                        current_stage = "B"
                        transitions.append("teleop->A(scripted)->B")
                        print(f"scripted Stage A 信息：{transfer_info}")
                        if not scripted_ok:
                            policy_obs = None
                            break

                if done and policy_obs is None:
                    print("手动阶段已结束，本回合未进入策略接管。")
                    break

            while policy_obs is not None:
                action, _ = models[current_stage].predict(policy_obs, deterministic=True)
                policy_obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                stage_rewards[current_stage] += float(reward)
                stage_steps[current_stage] += 1
                final_info = info

                if not args.no_render:
                    env.render()
                    time.sleep(args.sleep)

                if info.get("success", False):
                    if current_stage == "C":
                        success = True
                        break

                    next_stage = "B" if current_stage == "A" else "C"
                    flattened = capture_flattened_state(env)
                    env.close()
                    prev_stage = current_stage
                    current_stage = next_stage
                    env = create_stage_env(current_stage, object_profile, object_policy, vision_context, not args.no_render)
                    policy_obs, transfer_info = restore_stage(current_stage, env, flattened, prev_stage)
                    render_stage_intro(env, not args.no_render, args.sleep, args.stage_pause)
                    transitions.append(f"{prev_stage}->{current_stage}")
                    print(f"阶段切换：{prev_stage} 成功，进入 Stage {current_stage}，transfer 信息：{transfer_info}")
                    continue

                if terminated or truncated:
                    if current_stage == "C" and c_to_b_fallbacks < args.max_c_to_b_fallbacks:
                        c_to_b_fallbacks += 1
                        flattened = capture_flattened_state(env)
                        env.close()
                        prev_stage = current_stage
                        current_stage = "B"
                        env = create_stage_env(current_stage, object_profile, object_policy, vision_context, not args.no_render)
                        policy_obs, transfer_info = restore_stage(current_stage, env, flattened, prev_stage)
                        render_stage_intro(env, not args.no_render, args.sleep, args.stage_pause)
                        transitions.append(f"{prev_stage}->B(fallback)")
                        print(f"阶段回退：{prev_stage} 失败，回到 Stage B，transfer 信息：{transfer_info}")
                        continue

                    if current_stage == "B" and b_to_a_fallbacks < args.max_b_to_a_fallbacks:
                        b_to_a_fallbacks += 1
                        flattened = capture_flattened_state(env)
                        env.close()
                        prev_stage = current_stage
                        current_stage = "A"
                        env = create_stage_env(current_stage, object_profile, object_policy, vision_context, not args.no_render)
                        policy_obs, transfer_info = restore_stage(current_stage, env, flattened, prev_stage)
                        render_stage_intro(env, not args.no_render, args.sleep, args.stage_pause)
                        transitions.append(f"{prev_stage}->A(fallback)")
                        print(f"阶段回退：{prev_stage} 失败，回到 Stage A，transfer 信息：{transfer_info}")
                        continue
                    break

            print(
                f"结束：success={success}, manual_steps={manual_steps}, total_reward={total_reward:.2f}, "
                f"transitions={transitions}, stage_steps={stage_steps}, "
                f"stage_rewards={{{', '.join(f'{k}: {v:.2f}' for k, v in stage_rewards.items())}}}"
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
        finally:
            env.close()


if __name__ == "__main__":
    main()
