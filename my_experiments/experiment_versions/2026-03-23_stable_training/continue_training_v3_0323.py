"""
===============================================================================
        继续训练 PPO Lift 任务（从检查点恢复 + 改进超参数）
===============================================================================

创建日期：2026 年 3 月 23 日

用途：从之前的检查点继续训练，并改进超参数

改进内容：
1. 降低学习率：1e-4 → 5e-5
2. 增大 batch size：64 → 128
3. 增加 n_epochs：10 → 15

检查点位置：./ppo_checkpoints_v3/ppo_lift_v3_100000_steps.zip
目标步数：100 万步（从 10 万步继续到 100 万步，还需 90 万步）
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
import glob

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
RUN_NAME = f"lift_ppo_v3_continue_{RUN_TIMESTAMP}"
LEGACY_CHECKPOINT_DIR = "./ppo_checkpoints_v3"

# =============================================================================
# 环境类（必须与训练时完全相同）
# =============================================================================

print("=" * 70)
print("继续训练 PPO Lift 任务（改进超参数）")
print("=" * 70)
print("""
超参数改进：
1. 学习率：1e-4 → 5e-5（降低 50%，学习更稳定）
2. batch_size：64 → 128（更大的 batch，更稳定的梯度）
3. n_epochs：10 → 15（每次更新学习更充分）
""")

tracker = RunTracker(
    experiment_dir=SCRIPT_DIR,
    run_name=RUN_NAME,
    script_name=os.path.basename(__file__),
    purpose="稳定版继续训练",
)
print(f"本次继续训练目录：{tracker.run_dir}")


class RobosuiteGymWrapperStable(gym.Env):
    """
    带稳定奖励函数的 robosuite 环境包装器
    （必须与 lift_rl_training_v3_0323.py 中的类完全相同）
    """
    
    def __init__(self):
        super().__init__()
        
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


# =============================================================================
# 第 1 步：加载检查点
# =============================================================================

print("\n" + "=" * 70)
print("第 1 步：加载检查点")
print("=" * 70)

def extract_steps(filename):
    parts = os.path.basename(filename).split('_')
    # 找到包含数字的部分
    for part in parts:
        if part.isdigit():
            return int(part)
    return 0

search_dirs = []
if os.path.isdir(LEGACY_CHECKPOINT_DIR):
    search_dirs.append(LEGACY_CHECKPOINT_DIR)
search_dirs.extend(
    path for path in glob.glob("./training_runs/*/ppo_checkpoints_v3") if os.path.isdir(path)
)
search_dirs.extend(
    path for path in glob.glob("./training_runs/*/*/checkpoints") if os.path.isdir(path)
)

checkpoint_paths = []
for search_dir in search_dirs:
    checkpoint_paths.extend(glob.glob(os.path.join(search_dir, "*.zip")))

if not checkpoint_paths:
    print("❌ 没有找到可用的检查点文件")
    print(f"已搜索目录：{search_dirs}")
    exit()

checkpoint_paths.sort(key=lambda path: (extract_steps(path), os.path.getmtime(path)))
checkpoint_path = checkpoint_paths[-1]
latest_checkpoint = os.path.basename(checkpoint_path)

print(f"找到检查点：{latest_checkpoint}")
print(f"检查点路径：{checkpoint_path}")
print(f"加载模型中...")
tracker.record_base_checkpoint(checkpoint_path)

# 加载模型
model = PPO.load(checkpoint_path)
print("✅ 模型加载成功！")

# =============================================================================
# 第 2 步：创建环境
# =============================================================================

print("\n" + "=" * 70)
print("第 2 步：创建环境")
print("=" * 70)

def make_env():
    env = RobosuiteGymWrapperStable()
    env = Monitor(env, tracker.log_dir)
    return env

print("创建环境中...")
env = DummyVecEnv([make_env])
print("✅ 环境创建成功！")

# 设置模型的环境
model.set_env(env)

# =============================================================================
# 第 3 步：修改超参数
# =============================================================================

print("\n" + "=" * 70)
print("第 3 步：修改超参数")
print("=" * 70)

# 修改学习率
model.learning_rate = 5e-5
print(f"学习率：{model.learning_rate}（原 1e-4）")

# 修改 batch size
model.batch_size = 128
print(f"batch_size：{model.batch_size}（原 64）")

# 修改 n_epochs
model.n_epochs = 15
print(f"n_epochs：{model.n_epochs}（原 10）")

print("\n✅ 超参数已修改！")

# =============================================================================
# 第 4 步：继续训练
# =============================================================================

print("\n" + "=" * 70)
print("第 4 步：继续训练")
print("=" * 70)

# 计算还需要训练的步数
current_steps = extract_steps(checkpoint_path)
target_steps = 1000000
additional_steps = target_steps - current_steps

print(f"当前步数：{current_steps:,} 步")
print(f"目标步数：{target_steps:,} 步")
print(f"还需训练：{additional_steps:,} 步")

# 创建检查点回调
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
    # 继续训练
    model.learn(
        total_timesteps=additional_steps,
        callback=checkpoint_callback,
        tb_log_name=tracker.run_name,
        reset_num_timesteps=False,  # 保持步数连续
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
# 第 5 步：保存模型
# =============================================================================

print("\n" + "=" * 70)
print("第 5 步：保存模型")
print("=" * 70)

# 获取最终步数
final_steps = model.num_timesteps
if run_status == "completed":
    model_filename = f"final_model_steps_{final_steps}"
elif run_status == "interrupted":
    model_filename = f"interrupted_model_steps_{final_steps}"
else:
    model_filename = f"failed_model_steps_{final_steps}"

model_path = tracker.path_for(model_filename)
model.save(model_path)
print(f"✅ 模型已保存到：{model_path}.zip")

# 保存训练配置
config = {
    "创建日期": "2026-03-23",
    "版本": "v3 继续训练（改进超参数）",
    "从检查点恢复": latest_checkpoint,
    "检查点路径": checkpoint_path,
    "运行名称": tracker.run_name,
    "运行目录": tracker.run_dir,
    "状态": run_status,
    "初始步数": current_steps,
    "最终步数": final_steps,
    "训练步数": additional_steps,
    "输出目录": {
        "checkpoints": tracker.checkpoint_dir,
        "logs": tracker.log_dir,
        "tensorboard": tracker.tensorboard_dir,
    },
    "超参数改进": {
        "learning_rate": {"原": 0.0001, "新": 0.00005},
        "batch_size": {"原": 64, "新": 128},
        "n_epochs": {"原": 10, "新": 15}
    }
}

config_path = tracker.path_for("training_config.json")
with open(config_path, "w", encoding='utf-8') as f:
    json.dump(config, f, indent=2, ensure_ascii=False)
print(f"✅ 训练配置已保存到：{config_path}")

final_run_dir = tracker.finalize(
    status=run_status,
    final_steps=final_steps,
    artifacts={
        "model": f"{model_filename}.zip",
        "config": os.path.basename(config_path),
    },
    notes="稳定版继续训练",
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
   - {os.path.join(final_run_dir, f'{model_filename}.zip')}
   - {os.path.join(final_run_dir, 'checkpoints')}/
   - {os.path.join(final_run_dir, 'logs')}/
   - {os.path.join(final_run_dir, 'tensorboard')}/
   - {final_config_path}

📊 查看训练数据：
   cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
   tensorboard --logdir {os.path.join(final_run_dir, 'tensorboard')} --host 0.0.0.0 --port 6007

🎯 下一步：
   - 在 TensorBoard 中查看 value_loss 是否更稳定
   - 训练完成后测试模型
""")

if run_status == "failed":
    raise RuntimeError(error_message)
