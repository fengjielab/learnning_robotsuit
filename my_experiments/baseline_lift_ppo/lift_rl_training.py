"""
===============================================================================
                    使用 PPO 强化学习训练 Lift 任务（带 TensorBoard 监控）
===============================================================================

这个脚本会：
1. 创建 Lift 环境
2. 创建 PPO 模型（带 TensorBoard 日志）
3. 训练模型
4. 保存模型
5. 测试训练好的模型

训练过程中可以用 TensorBoard 实时监控！
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

# =============================================================================
# 第 1 步：创建 robosuite 环境并包装成 Gym 格式
# =============================================================================

print("=" * 70)
print("第 1 步：创建环境")
print("=" * 70)

class RobosuiteGymWrapper(gym.Env):
    """
    把 robosuite 环境包装成 Gym 格式，让 stable-baselines3 能用
    """
    def __init__(self):
        super().__init__()
        
        # 创建 robosuite 环境
        self.env = suite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,          # 训练时不显示
            has_offscreen_renderer=False,
            use_camera_obs=False,        # 不用图像观测（太慢）
            use_object_obs=True,         # 用物体观测
            reward_shaping=True,         # 用稠密奖励
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
        
        print(f"观测维度：{obs_dim}")
        print(f"动作维度：{self.env.action_dim}")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs = self.env.reset()
        # 把所有观测值拼成一个向量
        obs_vec = np.concatenate([
            np.array(obs[k]).flatten() 
            for k in obs.keys() 
            if isinstance(obs[k], np.ndarray)
        ]).astype(np.float32)
        return obs_vec, {}
    
    def step(self, action):
        """执行动作"""
        obs, reward, done, info = self.env.step(action)
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


# 创建日志目录
log_dir = "./ppo_logs"
os.makedirs(log_dir, exist_ok=True)

# 创建带监控的环境
def make_env():
    env = RobosuiteGymWrapper()
    env = Monitor(env, log_dir)  # 监控奖励等数据
    return env

print("创建环境中...")
env = DummyVecEnv([make_env])
print("环境创建成功！")

# =============================================================================
# 第 2 步：创建 PPO 模型（带 TensorBoard 日志）
# =============================================================================

print("\n" + "=" * 70)
print("第 2 步：创建 PPO 模型")
print("=" * 70)

# TensorBoard 日志目录
tb_log_dir = "./ppo_tensorboard"
os.makedirs(tb_log_dir, exist_ok=True)

model = PPO(
    "MlpPolicy",           # 使用 MLP 策略
    env,
    verbose=1,
    tensorboard_log=tb_log_dir,  # TensorBoard 日志路径
    learning_rate=3e-4,
    n_steps=2048,          # 每次更新的步数
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    ent_coef=0.01,         # 鼓励探索
)

print("PPO 模型创建成功！")
print(f"\n📊 TensorBoard 日志目录：{tb_log_dir}")

# =============================================================================
# 第 3 步：训练
# =============================================================================

print("\n" + "=" * 70)
print("第 3 步：开始训练")
print("=" * 70)
print("训练 10 万步（大约需要 10-30 分钟）...")
print("按 Ctrl+C 可以随时中断")

# 创建检查点回调（每 1 万步保存一次）
checkpoint_callback = CheckpointCallback(
    save_freq=10000,
    save_path="./ppo_checkpoints/",
    name_prefix="ppo_lift",
)

# 记录开始时间
start_time = time.time()

try:
    # 开始训练（带 TensorBoard 日志）
    model.learn(
        total_timesteps=100000,
        callback=checkpoint_callback,
        tb_log_name="lift_ppo_run1",
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n训练完成！耗时：{elapsed_time/60:.1f} 分钟")
    
except KeyboardInterrupt:
    print("\n训练被中断，但会保存当前模型...")

# =============================================================================
# 第 4 步：保存模型
# =============================================================================

print("\n" + "=" * 70)
print("第 4 步：保存模型")
print("=" * 70)

model.save("ppo_lift_final")
print("模型已保存到：ppo_lift_final.zip")

# 保存训练配置
config = {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "total_timesteps": 100000,
}
import json
with open("ppo_training_config.json", "w") as f:
    json.dump(config, f, indent=2)
print("训练配置已保存到：ppo_training_config.json")

# =============================================================================
# 第 5 步：测试训练好的模型
# =============================================================================

print("\n" + "=" * 70)
print("第 5 步：测试模型")
print("=" * 70)

# 重新加载模型
model = PPO.load("ppo_lift_final")

# 创建测试环境（带渲染）
test_env = RobosuiteGymWrapper()
test_env.env.has_renderer = True
test_env.env.has_offscreen_renderer = True

obs = test_env.reset()[0] if isinstance(test_env.reset(), tuple) else test_env.reset()
total_reward = 0

print("开始测试（按 Ctrl+C 停止）...")

try:
    for i in range(500):
        action, _ = model.predict(obs, deterministic=True)
        result = test_env.step(action)
        obs, reward, done, _, _ = result
        test_env.render()
        total_reward += reward
        
        if i % 50 == 0:
            print(f"步{i}: 奖励={reward:.4f}, 总奖励={total_reward:.4f}")
        
        if done:
            print("任务完成！")
            break

except KeyboardInterrupt:
    pass

print(f"\n测试完成！总奖励：{total_reward:.4f}")
test_env.env.close()

print("\n" + "=" * 70)
print("全部完成！")
print("=" * 70)
print("""
📁 生成的文件：
   - ppo_lift_final.zip      : 训练好的模型
   - ppo_checkpoints/        : 训练过程中的检查点
   - ppo_logs/               : 训练监控日志
   - ppo_tensorboard/        : TensorBoard 数据
   - ppo_training_config.json: 训练配置

📊 查看训练数据：

   1. 实时监控（训练时另开一个终端）：
      cd /home/mfj/robosuite/my_experiments/baseline_lift_ppo
      tensorboard --logdir ./ppo_tensorboard
      然后在浏览器打开 http://localhost:6006

   2. 查看奖励曲线：
      python plot_training_results.py

   3. 查看监控日志：
      cat ppo_logs/monitor.csv

下一步：
   - 修改训练参数重新训练
   - 增加训练步数到 50 万或 100 万
   - 调整网络结构
""")
