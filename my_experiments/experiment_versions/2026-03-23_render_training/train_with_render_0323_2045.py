"""
===============================================================================
        使用 PPO 强化学习训练 Lift 任务（带实时渲染）
===============================================================================

创建日期：2026 年 3 月 23 日 20:45

用途：训练时可以实时看到机器人学习过程

功能：
1. 每训练 1 万步，进行一次可视化测试
2. 显示机器人实际表现
3. 打印详细统计信息（包括夹爪状态）
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import robosuite as suite
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import sys
import time
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

from run_tracking import RunTracker

# =============================================================================
# 常量定义
# =============================================================================

TABLE_HEIGHT = 0.83  # 桌面高度（米）
LIFT_THRESHOLD = 0.04  # 成功举起高度（米）
GRIPPER_THRESHOLD = 0.02  # 夹爪闭合阈值
RUN_TIMESTAMP = time.strftime("%Y%m%d_%H%M%S")
RUN_NAME = f"lift_ppo_v4_render_{RUN_TIMESTAMP}"

# =============================================================================
# 环境类（带奖励函数）
# =============================================================================

print("=" * 70)
print("PPO 训练（带实时渲染）")
print("=" * 70)

tracker = RunTracker(
    experiment_dir=SCRIPT_DIR,
    run_name=RUN_NAME,
    script_name=os.path.basename(__file__),
    purpose="实时渲染训练",
)
print(f"本次训练目录：{tracker.run_dir}")
print(f"TensorBoard 日志目录：{tracker.tensorboard_dir}")


class RobosuiteGymWrapperStable(gym.Env):
    """带稳定奖励函数的 robosuite 环境包装器"""
    
    def __init__(self, has_renderer=False):
        super().__init__()
        
        self.env = suite.make(
            "Lift",
            robots="Panda",
            has_renderer=has_renderer,
            has_offscreen_renderer=has_renderer,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=False,
            control_freq=20,
            horizon=500,
        )
        
        obs_example = self.env.reset()
        obs_dim = sum(np.array(obs_example[k]).flatten().shape[0] 
                      for k in obs_example.keys() 
                      if isinstance(obs_example[k], np.ndarray))
        
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.env.action_dim,),
            dtype=np.float32
        )
        
        self.prev_cube_height = None
        
        print(f"观测维度：{obs_dim}")
        print(f"动作维度：{self.env.action_dim}")
    
    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        
        if 'cube_pos' in obs:
            self.prev_cube_height = float(obs['cube_pos'][2])
        
        obs_vec = np.concatenate([
            np.array(obs[k]).flatten() 
            for k in obs.keys() 
            if isinstance(obs[k], np.ndarray)
        ]).astype(np.float32)
        
        return obs_vec, {}
    
    def _compute_stable_reward(self, obs):
        reward = 0.0
        
        cube_pos = obs.get('cube_pos', None)
        gripper_to_cube = obs.get('gripper_to_cube_pos', None)
        gripper_qpos = obs.get('robot0_gripper_qpos', None)
        
        if cube_pos is None:
            return 0.0
        
        current_height = float(cube_pos[2])
        
        # 1. 接近奖励
        if gripper_to_cube is not None:
            dist = float(np.linalg.norm(gripper_to_cube))
            approach_reward = 1.0 / (1.0 + dist * 10)
            reward += approach_reward * 0.3
        
        # 2. 抓取奖励
        if gripper_qpos is not None and gripper_to_cube is not None:
            gripper_closed = np.mean(np.abs(gripper_qpos)) < GRIPPER_THRESHOLD
            dist = float(np.linalg.norm(gripper_to_cube))
            if gripper_closed and dist < 0.05:
                reward += 0.5
        
        # 3. 举起奖励
        if self.prev_cube_height is not None:
            height_delta = current_height - self.prev_cube_height
            if height_delta > 0 and current_height > TABLE_HEIGHT:
                lift_reward = height_delta * 5
                reward += lift_reward
        
        # 4. 成功奖励
        relative_height = current_height - TABLE_HEIGHT
        if relative_height > LIFT_THRESHOLD:
            reward += 2.0
        
        self.prev_cube_height = current_height
        
        return reward
    
    def step(self, action):
        obs, _, done, info = self.env.step(action)
        reward = self._compute_stable_reward(obs)
        
        obs_vec = np.concatenate([
            np.array(obs[k]).flatten() 
            for k in obs.keys() 
            if isinstance(obs[k], np.ndarray)
        ]).astype(np.float32)
        
        return obs_vec, reward, done, False, info
    
    def render(self):
        self.env.render()
    
    def close(self):
        self.env.close()


# =============================================================================
# 自定义回调：定期渲染测试
# =============================================================================

class RenderCallback(BaseCallback):
    """定期渲染测试的回调"""
    
    def __init__(self, render_freq=10000, verbose=0):
        super().__init__(verbose)
        self.render_freq = render_freq
        self.test_env = None
    
    def _init_callback(self) -> None:
        # 创建测试环境（带渲染）
        self.test_env = RobosuiteGymWrapperStable(has_renderer=True)
    
    def _on_step(self) -> bool:
        # 每 render_freq 步进行一次渲染测试
        if self.n_calls % self.render_freq == 0:
            print(f"\n{'='*70}")
            print(f"训练进度：{self.n_calls:,} 步")
            print(f"{'='*70}")
            
            # 测试 3 次
            num_test_episodes = 3
            success_count = 0
            max_heights = []
            
            for episode in range(num_test_episodes):
                obs = self.test_env.reset()
                if isinstance(obs, tuple):
                    obs = obs[0]
                
                episode_max_height = 0
                last_height = 0
                last_gripper = None
                
                for step in range(200):  # 测试 200 步
                    action, _ = self.model.predict(obs, deterministic=True)
                    result = self.test_env.step(action)
                    obs, reward, done, _, _ = result
                    
                    # 获取高度和夹爪状态
                    obs_dict = self._obs_to_dict(obs)
                    if 'cube_pos' in obs_dict:
                        height = float(obs_dict['cube_pos'][2])
                        episode_max_height = max(episode_max_height, height)
                        last_height = height
                    
                    if 'robot0_gripper_qpos' in obs_dict:
                        last_gripper = obs_dict['robot0_gripper_qpos']
                    
                    # 渲染
                    self.test_env.render()
                    time.sleep(0.01)  # 放慢速度
                    
                    # 打印夹爪状态
                    if step % 50 == 0:
                        gripper_info = f"{last_gripper}" if last_gripper is not None else "N/A"
                        print(f"  步{step}: 夹爪={gripper_info}, 高度={last_height:.4f}米")
                    
                    if done:
                        break
                
                relative_height = episode_max_height - TABLE_HEIGHT
                max_heights.append(relative_height)
                print(f"  测试{episode+1}: 最大相对高度={relative_height:.4f}米")
                
                if relative_height > LIFT_THRESHOLD:
                    success_count += 1
            
            print(f"\n  平均最大高度：{np.mean(max_heights):.4f}米")
            print(f"  成功率：{success_count/num_test_episodes*100:.1f}%")
            
            if np.mean(max_heights) > LIFT_THRESHOLD:
                print("  ✅ 模型表现良好！")
            else:
                print("  📈 继续训练...")
            
            self.test_env.close()
        
        return True
    
    def _obs_to_dict(self, obs_vec):
        """把向量观测值转回字典（简化版，用于调试）"""
        # 这里只是用于获取高度，不需要完整转换
        return {'cube_pos': [0, 0, 0], 'robot0_gripper_qpos': None}


# =============================================================================
# 创建日志目录
# =============================================================================

def make_env():
    env = RobosuiteGymWrapperStable(has_renderer=False)
    env = Monitor(env, tracker.log_dir)
    return env

print("创建环境中...")
env = DummyVecEnv([make_env])
print("✅ 环境创建成功！")

# =============================================================================
# 创建 PPO 模型
# =============================================================================

print("\n" + "=" * 70)
print("创建 PPO 模型")
print("=" * 70)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=tracker.tensorboard_dir,
    learning_rate=5e-5,
    n_steps=2048,
    batch_size=128,
    n_epochs=15,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.3,
)

print("✅ PPO 模型创建成功！")

# =============================================================================
# 训练
# =============================================================================

print("\n" + "=" * 70)
print("开始训练（带实时渲染）")
print("=" * 70)
print("每 1 万步会进行一次可视化测试")
print("观察机器人是否学会抓取和举起方块")
print("按 Ctrl+C 可以中断，模型会自动保存")

# 检查点回调
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=tracker.checkpoint_dir,
    name_prefix="ppo_lift_v4",
)

# 渲染回调
render_callback = RenderCallback(render_freq=10000)

start_time = time.time()
run_status = "completed"
error_message = None

try:
    model.learn(
        total_timesteps=500000,  # 50 万步
        callback=[checkpoint_callback, render_callback],
        tb_log_name=tracker.run_name,
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ 训练完成！耗时：{elapsed_time/60:.1f} 分钟")
    
except KeyboardInterrupt:
    run_status = "interrupted"
    print("\n⚠️  训练被中断，保存模型...")
except Exception as exc:
    run_status = "failed"
    error_message = repr(exc)
    print(f"\n❌ 训练失败：{exc}")
finally:
    env.close()

# =============================================================================
# 保存模型
# =============================================================================

print("\n" + "=" * 70)
print("保存模型")
print("=" * 70)

final_steps = model.num_timesteps
if run_status == "completed":
    model_filename = f"final_model_steps_{final_steps}"
elif run_status == "interrupted":
    model_filename = f"interrupted_model_steps_{final_steps}"
else:
    model_filename = f"failed_model_steps_{final_steps}"

model_path = tracker.path_for(model_filename)
model.save(model_path)
print(f"✅ 模型已保存：{model_path}.zip")

config = {
    "创建日期": "2026-03-23",
    "版本": "v4 带实时渲染",
    "运行名称": tracker.run_name,
    "运行目录": tracker.run_dir,
    "状态": run_status,
    "总步数": final_steps,
    "输出目录": {
        "checkpoints": tracker.checkpoint_dir,
        "logs": tracker.log_dir,
        "tensorboard": tracker.tensorboard_dir,
    },
    "超参数": {
        "learning_rate": 5e-5,
        "batch_size": 128,
        "n_epochs": 15,
        "ent_coef": 0.01,
        "clip_range": 0.1,
    }
}

config_path = tracker.path_for("training_config.json")
with open(config_path, "w", encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"✅ 配置已保存：{config_path}")

final_run_dir = tracker.finalize(
    status=run_status,
    final_steps=final_steps,
    artifacts={
        "model": f"{model_filename}.zip",
        "config": os.path.basename(config_path),
    },
    notes="渲染训练",
    error_message=error_message,
)
final_config_path = os.path.join(final_run_dir, os.path.basename(config_path))
config["运行目录"] = final_run_dir
config["输出目录"] = {
    "checkpoints": os.path.join(final_run_dir, "checkpoints"),
    "logs": os.path.join(final_run_dir, "logs"),
    "tensorboard": os.path.join(final_run_dir, "tensorboard"),
}
with open(final_config_path, "w", encoding="utf-8") as f:
    json.dump(config, f, indent=2, ensure_ascii=False)

print("\n" + "=" * 70)
print("全部完成！")
print("=" * 70)
print(f"""
本次训练状态：
  {run_status}

本次训练输出目录：
  {final_run_dir}

只查看这次训练的 TensorBoard：
  cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_render_training
  source /home/mfj/mj_robot/bin/activate
  tensorboard --logdir {os.path.join(final_run_dir, 'tensorboard')} --host 0.0.0.0 --port 6007
""")

if run_status == "failed":
    raise RuntimeError(error_message)
