"""
===============================================================================
        使用 PPO 强化学习训练 Lift 任务（改进版 v2 - 自定义奖励函数）
===============================================================================

创建日期：2026 年 3 月 23 日

改进内容：
1. 自定义奖励函数（增加抓取和举起奖励）
2. 更多训练步数（100 万步）
3. 更好的超参数
4. 每 10 万步保存一次检查点

训练目标：
- 让机器人学会抓取并举起方块
- 成功标准：方块相对桌面高度 > 0.04 米
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
import robosuite as suite
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import os
import time
import json

# =============================================================================
# 第 1 步：创建 robosuite 环境（带自定义奖励函数）
# =============================================================================

print("=" * 70)
print("第 1 步：创建环境（带自定义奖励函数）")
print("=" * 70)

# 常量定义
TABLE_HEIGHT = 0.83  # 桌面高度（米）
LIFT_THRESHOLD = 0.04  # 成功举起高度（米）
GRIPPER_THRESHOLD = 0.02  # 夹爪闭合阈值


class RobosuiteGymWrapperWithReward(gym.Env):
    """
    带自定义奖励函数的 robosuite 环境包装器
    
    奖励设计：
    1. 接近奖励：鼓励手臂接近方块（权重降低）
    2. 抓取奖励：当夹爪闭合且接近方块时给予奖励
    3. 举起奖励：当方块被举起时给予大量奖励
    4. 成功奖励：当方块举起超过阈值时给予额外奖励
    """
    
    def __init__(self):
        super().__init__()
        
        # 创建 robosuite 环境（关闭默认奖励）
        self.env = suite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=False,  # 关闭默认奖励 shaping，使用自定义奖励
            control_freq=20,
            horizon=500,
        )
        
        # 获取观测空间维度
        obs_example = self.env.reset()
        obs_dim = sum(np.array(obs_example[k]).flatten().shape[0] 
                      for k in obs_example.keys() 
                      if isinstance(obs_example[k], np.ndarray))
        
        # 定义观测空间
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # 定义动作空间
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(self.env.action_dim,),
            dtype=np.float32
        )
        
        # 记录上一步的状态
        self.prev_cube_height = None
        self.prev_gripper_dist = None
        
        print(f"观测维度：{obs_dim}")
        print(f"动作维度：{self.env.action_dim}")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs = self.env.reset()
        
        # 初始化记录
        if 'cube_pos' in obs:
            self.prev_cube_height = float(obs['cube_pos'][2])
        if 'gripper_to_cube_pos' in obs:
            gripper_to_cube = obs['gripper_to_cube_pos']
            self.prev_gripper_dist = float(np.linalg.norm(gripper_to_cube))
        
        # 把观测值拼成向量
        obs_vec = np.concatenate([
            np.array(obs[k]).flatten() 
            for k in obs.keys() 
            if isinstance(obs[k], np.ndarray)
        ]).astype(np.float32)
        
        return obs_vec, {}
    
    def _compute_custom_reward(self, obs, action):
        """
        计算自定义奖励
        
        奖励组成：
        1. 接近奖励：-distance_to_cube（最大 1）
        2. 抓取奖励：当夹爪闭合且距离近时 +1
        3. 举起奖励：(current_height - initial_height) * 10
        4. 成功奖励：当高度 > 阈值时 +5
        """
        reward = 0.0
        
        # 获取当前状态
        cube_pos = obs.get('cube_pos', None)
        gripper_to_cube = obs.get('gripper_to_cube_pos', None)
        gripper_qpos = obs.get('robot0_gripper_qpos', None)
        
        if cube_pos is None:
            return 0.0
        
        current_height = float(cube_pos[2])
        
        # 1. 接近奖励（权重降低）
        if gripper_to_cube is not None:
            dist = float(np.linalg.norm(gripper_to_cube))
            approach_reward = 1.0 / (1.0 + dist * 10)  # 距离越近奖励越大，最大 1
            reward += approach_reward * 0.5  # 降低权重
        
        # 2. 抓取奖励
        if gripper_qpos is not None and gripper_to_cube is not None:
            gripper_closed = np.mean(np.abs(gripper_qpos)) < GRIPPER_THRESHOLD
            dist = float(np.linalg.norm(gripper_to_cube))
            if gripper_closed and dist < 0.05:
                reward += 2.0  # 抓取成功奖励
        
        # 3. 举起奖励（主要奖励来源）
        if self.prev_cube_height is not None:
            height_delta = current_height - self.prev_cube_height
            # 只有当方块被举起时才给奖励
            if height_delta > 0 and current_height > TABLE_HEIGHT:
                lift_reward = height_delta * 20  # 增加权重
                reward += lift_reward
        
        # 4. 成功奖励
        relative_height = current_height - TABLE_HEIGHT
        if relative_height > LIFT_THRESHOLD:
            reward += 10.0  # 成功举起的大量奖励
        
        # 5. 稀疏任务完成奖励（来自环境）
        try:
            if hasattr(self.env, '_check_success'):
                if self.env._check_success():
                    reward += 5.0
        except:
            pass
        
        # 更新记录
        self.prev_cube_height = current_height
        if gripper_to_cube is not None:
            self.prev_gripper_dist = float(np.linalg.norm(gripper_to_cube))
        
        return reward
    
    def step(self, action):
        """执行动作并返回自定义奖励"""
        obs, _, done, info = self.env.step(action)
        
        # 计算自定义奖励
        reward = self._compute_custom_reward(obs, action)
        
        # 把观测值拼成向量
        obs_vec = np.concatenate([
            np.array(obs[k]).flatten() 
            for k in obs.keys() 
            if isinstance(obs[k], np.ndarray)
        ]).astype(np.float32)
        
        return obs_vec, reward, done, False, info
    
    def render(self):
        """渲染"""
        self.env.render()


# =============================================================================
# 创建日志目录
# =============================================================================

log_dir = "./ppo_logs_v2"
os.makedirs(log_dir, exist_ok=True)

# 创建带监控的环境
def make_env():
    env = RobosuiteGymWrapperWithReward()
    env = Monitor(env, log_dir)
    return env

print("创建环境中...")
env = DummyVecEnv([make_env])
print("✅ 环境创建成功！")

# =============================================================================
# 第 2 步：创建 PPO 模型（改进的超参数）
# =============================================================================

print("\n" + "=" * 70)
print("第 2 步：创建 PPO 模型（改进的超参数）")
print("=" * 70)

# TensorBoard 日志目录
tb_log_dir = "./ppo_tensorboard_v2"
os.makedirs(tb_log_dir, exist_ok=True)

# 改进的超参数
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=tb_log_dir,
    learning_rate=3e-4,      # 保持原值
    n_steps=2048,            # 保持原值
    batch_size=128,          # 增加 batch size
    n_epochs=20,             # 增加 epoch 数，学习更充分
    gamma=0.99,              # 保持原值
    gae_lambda=0.95,         # 保持原值
    clip_range=0.2,          # 保持原值
    ent_coef=0.03,           # 增加探索率（原 0.01）
    vf_coef=0.5,             # 价值函数权重
    max_grad_norm=0.5,       # 梯度裁剪
)

print("✅ PPO 模型创建成功！")
print(f"\n📊 TensorBoard 日志目录：{tb_log_dir}")

# =============================================================================
# 第 3 步：训练（100 万步）
# =============================================================================

print("\n" + "=" * 70)
print("第 3 步：开始训练")
print("=" * 70)
print("训练目标：100 万步（大约需要 2-5 小时）")
print("按 Ctrl+C 可以随时中断，模型会自动保存")

# 创建检查点回调（每 10 万步保存一次）
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path="./ppo_checkpoints_v2/",
    name_prefix="ppo_lift_v2",
)

# 记录开始时间
start_time = time.time()

try:
    # 开始训练
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        tb_log_name="lift_ppo_v2",
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ 训练完成！耗时：{elapsed_time/60:.1f} 分钟")
    
except KeyboardInterrupt:
    print("\n⚠️  训练被中断，但会保存当前模型...")

# =============================================================================
# 第 4 步：保存模型
# =============================================================================

print("\n" + "=" * 70)
print("第 4 步：保存模型")
print("=" * 70)

model.save("ppo_lift_final_v4_0323")
print("✅ 模型已保存到：ppo_lift_final_v4_0323.zip")

# 保存训练配置
config = {
    "创建日期": "2026-03-23",
    "改进内容": [
        "自定义奖励函数",
        "增加抓取和举起奖励权重",
        "增加探索率",
        "增加 batch size 和 epochs"
    ],
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 128,
    "n_epochs": 20,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.03,
    "total_timesteps": 1000000,
    "奖励设计": {
        "接近奖励": "距离越近奖励越大，最大 0.5",
        "抓取奖励": "夹爪闭合且距离近时 +2",
        "举起奖励": "高度增量 × 20",
        "成功奖励": "高度超过阈值时 +10"
    }
}

with open("ppo_training_config_v2_0323.json", "w", encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print("✅ 训练配置已保存到：ppo_training_config_v2_0323.json")

# =============================================================================
# 第 5 步：简单测试
# =============================================================================

print("\n" + "=" * 70)
print("第 5 步：快速测试模型")
print("=" * 70)

# 重新加载模型
model = PPO.load("ppo_lift_final_v4_0323")

# 创建测试环境（带渲染）
test_env = RobosuiteGymWrapperWithReward()
test_env.env.has_renderer = True
test_env.env.has_offscreen_renderer = True

obs = test_env.reset()
if isinstance(obs, tuple):
    obs = obs[0]

total_reward = 0
max_height = 0

print("开始测试（500 步，按 Ctrl+C 停止）...")

try:
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        result = test_env.step(action)
        obs, reward, done, _, _ = result
        test_env.render()
        total_reward += reward
        
        # 获取方块高度
        obs_dict = {k: v for k, v in zip(
            ['robot0_joint_pos_cos', 'robot0_joint_pos_sin', 'robot0_joint_vel',
             'robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', 
             'robot0_gripper_qvel', 'cube_pos', 'cube_quat', 'gripper_to_cube_pos',
             'robot0_proprio-state', 'object-state'],
            [obs[:7], obs[7:14], obs[14:21], obs[21:24], obs[24:28], obs[28:30],
             obs[30:32], obs[32:35], obs[35:39], obs[39:42], obs[42:74], obs[74:84]]
        )}
        if 'cube_pos' in obs_dict:
            height = float(obs_dict['cube_pos'][2])
            max_height = max(max_height, height)
        
        if i % 50 == 0:
            print(f"步{i}: 奖励={reward:.4f}, 总奖励={total_reward:.4f}, 最大高度={max_height:.4f}米")
        
        if done:
            print("✅ 任务完成！")
            break

except KeyboardInterrupt:
    pass

relative_height = max_height - TABLE_HEIGHT
print(f"\n测试结果:")
print(f"  总奖励：{total_reward:.4f}")
print(f"  最大高度：{max_height:.4f}米")
print(f"  最大相对高度：{relative_height:.4f}米")
print(f"  成功标准：>{LIFT_THRESHOLD}米")
if relative_height > LIFT_THRESHOLD:
    print("  ✅ 达到成功标准！")
else:
    print(f"  ❌ 未达到标准，还差 {LIFT_THRESHOLD - relative_height:.4f}米")

test_env.env.close()

# =============================================================================
# 完成
# =============================================================================

print("\n" + "=" * 70)
print("全部完成！")
print("=" * 70)
print("""
📁 生成的文件：
   - ppo_lift_final_v4_0323.zip      : 训练好的模型
   - ppo_checkpoints_v2/             : 训练检查点（每 10 万步）
   - ppo_logs_v2/                    : 训练监控日志
   - ppo_tensorboard_v2/             : TensorBoard 数据
   - ppo_training_config_v2_0323.json: 训练配置

📊 查看训练数据：
   1. 实时监控：
      cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
      tensorboard --logdir ./ppo_tensorboard_v2
      
   2. 检查成功标准：
      python check_success_0323.py
      
   3. 完整测试模型：
      python test_model_0323.py

🎯 下一步：
   - 观察 TensorBoard 监控训练
   - 训练完成后测试模型
   - 如果效果不好，继续调整奖励函数
""")
