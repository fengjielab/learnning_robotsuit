"""
===============================================================================
                    继续训练已保存的 PPO 模型（50 万步版本）
===============================================================================
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import robosuite as suite
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os

# =============================================================================
# 环境类（和之前一样）
# =============================================================================

class RobosuiteGymWrapper(gym.Env):
    def __init__(self):
        super().__init__()
        
        self.env = suite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=True,
            control_freq=20,
            horizon=500,
        )
        
        obs_example = self.env.reset()
        obs_dim = sum(np.array(obs_example[k]).flatten().shape[0] 
                      for k in obs_example.keys() 
                      if isinstance(obs_example[k], np.ndarray))
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.env.action_dim,), dtype=np.float32
        )
        
        print(f"观测维度：{obs_dim}")
        print(f"动作维度：{self.env.action_dim}")
    
    def reset(self, seed=None, options=None):
        obs = self.env.reset()
        obs_vec = np.concatenate([
            np.array(obs[k]).flatten() 
            for k in obs.keys() 
            if isinstance(obs[k], np.ndarray)
        ]).astype(np.float32)
        return obs_vec, {}
    
    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs_vec = np.concatenate([
            np.array(obs[k]).flatten() 
            for k in obs.keys() 
            if isinstance(obs[k], np.ndarray)
        ]).astype(np.float32)
        return obs_vec, reward, done, False, info
    
    def render(self):
        self.env.render()


# =============================================================================
# 第 1 步：加载已训练的模型
# =============================================================================

print("=" * 70)
print("第 1 步：加载已训练的模型")
print("=" * 70)

# 加载最新的模型（v2 版本）
model_path = "ppo_lift_final_v2.zip"
model = PPO.load(model_path)

print(f"✅ 模型加载成功：{model_path}")

# =============================================================================
# 第 2 步：创建环境并设置到模型
# =============================================================================

print("\n" + "=" * 70)
print("第 2 步：创建环境")
print("=" * 70)

env = DummyVecEnv([lambda: RobosuiteGymWrapper()])
model.set_env(env)

print("✅ 环境设置成功！")

# =============================================================================
# 第 3 步：继续训练
# =============================================================================

print("\n" + "=" * 70)
print("第 3 步：继续训练")
print("=" * 70)

# 训练参数：50 万步
additional_timesteps = 500000  # 再训练 50 万步

print(f"将继续训练 {additional_timesteps} 步（约 50 万步）...")
print("预计时间：1-3 小时")
print("按 Ctrl+C 可以随时中断")

# 检查点回调（每 5 万步保存一次）
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path="./ppo_checkpoints_continue/",
    name_prefix="ppo_lift_continue",
)

try:
    model.learn(
        total_timesteps=additional_timesteps,
        callback=checkpoint_callback,
        tb_log_name="lift_ppo_continue_run2",
        reset_num_timesteps=False,  # 保持步数连续
    )
    print("\n✅ 继续训练完成！")
except KeyboardInterrupt:
    print("\n训练被中断，但会保存当前模型...")

# =============================================================================
# 第 4 步：保存模型
# =============================================================================

print("\n" + "=" * 70)
print("第 4 步：保存模型")
print("=" * 70)

model.save("ppo_lift_final_v3")
print("模型已保存到：ppo_lift_final_v3.zip")

print("\n" + "=" * 70)
print("全部完成！")
print("=" * 70)
print("""
生成的文件：
- ppo_lift_final_v3.zip     : 继续训练后的模型（v3 版本）
- ppo_checkpoints_continue/ : 继续训练的检查点

训练总步数：
- 第 1 次训练：10 万步
- 第 2 次训练：10 万步
- 第 3 次训练：50 万步
- 总计：约 70 万步
""")