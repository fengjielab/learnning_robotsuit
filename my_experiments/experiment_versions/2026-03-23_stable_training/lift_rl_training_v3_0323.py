"""
===============================================================================
        使用 PPO 强化学习训练 Lift 任务（稳定版 v3 - 修复训练不稳定问题）
===============================================================================

创建日期：2026 年 3 月 23 日

问题分析：
- v2 版本训练不稳定：std 过高 (20+)，value_loss 波动大
- 原因：探索率太高，学习率太高

改进内容：
1. 降低探索率 (ent_coef: 0.03 → 0.01)
2. 降低学习率 (3e-4 → 1e-4)
3. 更保守的 clip_range (0.2 → 0.1)
4. 更严格的梯度裁剪 (0.5 → 0.3)
5. 简化奖励函数，避免奖励爆炸
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
RUN_NAME = f"lift_ppo_v3_{RUN_TIMESTAMP}"

# =============================================================================
# 第 1 步：创建 robosuite 环境（带简化的奖励函数）
# =============================================================================

print("=" * 70)
print("第 1 步：创建环境（稳定版奖励函数）")
print("=" * 70)

tracker = RunTracker(
    experiment_dir=SCRIPT_DIR,
    run_name=RUN_NAME,
    script_name=os.path.basename(__file__),
    purpose="稳定版从头训练",
)
print(f"本次训练目录：{tracker.run_dir}")


class RobosuiteGymWrapperStable(gym.Env):
    """
    带稳定奖励函数的 robosuite 环境包装器
    
    奖励设计（简化版）：
    1. 接近奖励：-distance_to_cube（最大 0.5）
    2. 抓取奖励：当夹爪闭合且接近时 +1
    3. 举起奖励：高度增量 × 5（降低权重）
    4. 成功奖励：当高度 > 阈值时 +2（降低权重）
    """
    
    def __init__(self):
        super().__init__()
        
        # 创建 robosuite 环境（关闭默认奖励 shaping）
        self.env = suite.make(
            "Lift",
            robots="Panda",
            has_renderer=False,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            use_object_obs=True,
            reward_shaping=False,
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
        
        print(f"观测维度：{obs_dim}")
        print(f"动作维度：{self.env.action_dim}")
    
    def reset(self, seed=None, options=None):
        """重置环境"""
        obs = self.env.reset()
        
        # 初始化记录
        if 'cube_pos' in obs:
            self.prev_cube_height = float(obs['cube_pos'][2])
        
        # 把观测值拼成向量
        obs_vec = np.concatenate([
            np.array(obs[k]).flatten() 
            for k in obs.keys() 
            if isinstance(obs[k], np.ndarray)
        ]).astype(np.float32)
        
        return obs_vec, {}
    
    def _compute_stable_reward(self, obs):
        """
        计算稳定的奖励（简化版，避免奖励爆炸）
        """
        reward = 0.0
        
        # 获取当前状态
        cube_pos = obs.get('cube_pos', None)
        gripper_to_cube = obs.get('gripper_to_cube_pos', None)
        gripper_qpos = obs.get('robot0_gripper_qpos', None)
        
        if cube_pos is None:
            return 0.0
        
        current_height = float(cube_pos[2])
        
        # 1. 接近奖励（降低权重）
        if gripper_to_cube is not None:
            dist = float(np.linalg.norm(gripper_to_cube))
            approach_reward = 1.0 / (1.0 + dist * 10)
            reward += approach_reward * 0.3  # 降低权重
        
        # 2. 抓取奖励（降低权重）
        if gripper_qpos is not None and gripper_to_cube is not None:
            gripper_closed = np.mean(np.abs(gripper_qpos)) < GRIPPER_THRESHOLD
            dist = float(np.linalg.norm(gripper_to_cube))
            if gripper_closed and dist < 0.05:
                reward += 0.5  # 降低权重
        
        # 3. 举起奖励（降低权重）
        if self.prev_cube_height is not None:
            height_delta = current_height - self.prev_cube_height
            if height_delta > 0 and current_height > TABLE_HEIGHT:
                lift_reward = height_delta * 5  # 降低权重
                reward += lift_reward
        
        # 4. 成功奖励（降低权重）
        relative_height = current_height - TABLE_HEIGHT
        if relative_height > LIFT_THRESHOLD:
            reward += 2.0  # 降低权重
        
        # 更新记录
        self.prev_cube_height = current_height
        
        return reward
    
    def step(self, action):
        """执行动作并返回奖励"""
        obs, _, done, info = self.env.step(action)
        
        # 计算奖励
        reward = self._compute_stable_reward(obs)
        
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

# 创建带监控的环境
def make_env():
    env = RobosuiteGymWrapperStable()
    env = Monitor(env, tracker.log_dir)
    return env

print("创建环境中...")
env = DummyVecEnv([make_env])
print("✅ 环境创建成功！")

# =============================================================================
# 第 2 步：创建 PPO 模型（稳定的超参数）
# =============================================================================

print("\n" + "=" * 70)
print("第 2 步：创建 PPO 模型（稳定版超参数）")
print("=" * 70)

# 稳定的超参数
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log=tracker.tensorboard_dir,
    learning_rate=1e-4,      # 降低学习率（原 3e-4）
    n_steps=2048,
    batch_size=64,           # 恢复原值
    n_epochs=10,             # 恢复原值
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.1,          # 更保守的更新（原 0.2）
    ent_coef=0.01,           # 降低探索率（原 0.03）
    vf_coef=0.5,
    max_grad_norm=0.3,       # 更严格的梯度裁剪（原 0.5）
)

print("✅ PPO 模型创建成功！")
print(f"\n📊 TensorBoard 日志目录：{tracker.tensorboard_dir}")

# =============================================================================
# 第 3 步：训练（100 万步）
# =============================================================================

print("\n" + "=" * 70)
print("第 3 步：开始训练（稳定版）")
print("=" * 70)
print("训练目标：100 万步（大约需要 2-5 小时）")
print("按 Ctrl+C 可以随时中断，模型会自动保存")

# 创建检查点回调（每 10 万步保存一次）
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=tracker.checkpoint_dir,
    name_prefix="ppo_lift_v3",
)

# 记录开始时间
start_time = time.time()
run_status = "completed"
error_message = None

try:
    # 开始训练
    model.learn(
        total_timesteps=1000000,
        callback=checkpoint_callback,
        tb_log_name=tracker.run_name,
    )
    
    elapsed_time = time.time() - start_time
    print(f"\n✅ 训练完成！耗时：{elapsed_time/60:.1f} 分钟")
    
except KeyboardInterrupt:
    run_status = "interrupted"
    print("\n⚠️  训练被中断，但会保存当前模型...")
except Exception as exc:
    run_status = "failed"
    error_message = repr(exc)
    print(f"\n❌ 训练失败：{exc}")
finally:
    env.close()

# =============================================================================
# 第 4 步：保存模型
# =============================================================================

print("\n" + "=" * 70)
print("第 4 步：保存模型")
print("=" * 70)

if run_status == "completed":
    model_filename = f"final_model_steps_{model.num_timesteps}"
elif run_status == "interrupted":
    model_filename = f"interrupted_model_steps_{model.num_timesteps}"
else:
    model_filename = f"failed_model_steps_{model.num_timesteps}"

model_path = tracker.path_for(model_filename)
model.save(model_path)
print(f"✅ 模型已保存到：{model_path}.zip")

# 保存训练配置
config = {
    "创建日期": "2026-03-23",
    "版本": "v3 稳定版",
    "运行名称": tracker.run_name,
    "运行目录": tracker.run_dir,
    "状态": run_status,
    "输出目录": {
        "checkpoints": tracker.checkpoint_dir,
        "logs": tracker.log_dir,
        "tensorboard": tracker.tensorboard_dir,
    },
    "改进内容": [
        "降低探索率 (0.03 → 0.01)",
        "降低学习率 (3e-4 → 1e-4)",
        "更保守的 clip_range (0.2 → 0.1)",
        "更严格的梯度裁剪 (0.5 → 0.3)",
        "简化奖励函数，避免奖励爆炸"
    ],
    "超参数对比": {
        "learning_rate": {"v2": 0.0003, "v3": 0.0001},
        "ent_coef": {"v2": 0.03, "v3": 0.01},
        "clip_range": {"v2": 0.2, "v3": 0.1},
        "max_grad_norm": {"v2": 0.5, "v3": 0.3}
    },
    "learning_rate": 1e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "total_timesteps": 1000000,
}

config_path = tracker.path_for("training_config.json")
with open(config_path, "w", encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"✅ 训练配置已保存到：{config_path}")

# =============================================================================
# 第 5 步：快速测试
# =============================================================================

if run_status != "failed":
    print("\n" + "=" * 70)
    print("第 5 步：快速测试模型")
    print("=" * 70)

    # 重新加载模型
    model = PPO.load(model_path)

    # 创建测试环境（带渲染）
    test_env = RobosuiteGymWrapperStable()
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
else:
    print("\n跳过快速测试：训练过程失败，保留当前输出供排查。")

final_run_dir = tracker.finalize(
    status=run_status,
    final_steps=model.num_timesteps,
    artifacts={
        "model": f"{model_filename}.zip",
        "config": os.path.basename(config_path),
    },
    notes="稳定版从头训练",
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

# =============================================================================
# 完成
# =============================================================================

print("\n" + "=" * 70)
print("全部完成！")
print("=" * 70)
print(f"""
📁 生成的文件：
   - {os.path.join(final_run_dir, f'{model_filename}.zip')} : 训练好的模型
   - {os.path.join(final_run_dir, 'checkpoints')}/          : 训练检查点（每 10 万步）
   - {os.path.join(final_run_dir, 'logs')}/                 : 训练监控日志
   - {os.path.join(final_run_dir, 'tensorboard')}/          : TensorBoard 数据
   - {final_config_path} : 训练配置

📊 查看训练数据：
   1. 实时监控：
      cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
      tensorboard --logdir {os.path.join(final_run_dir, 'tensorboard')} --host 0.0.0.0 --port 6007
      
   2. 检查成功标准：
      python check_success_0323.py
      
   3. 完整测试模型：
      python test_model_0323.py

🎯 下一步：
   - 观察 TensorBoard 监控训练
   - 检查 std 和 value_loss 是否稳定
   - 训练完成后测试模型
""")

if run_status == "failed":
    raise RuntimeError(error_message)
