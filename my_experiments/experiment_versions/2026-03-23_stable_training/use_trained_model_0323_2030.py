"""
===============================================================================
                     使用已训练好的 PPO 模型
===============================================================================

创建日期：2026 年 3 月 23 日 20:30

用途：加载并使用已训练好的模型进行测试

模型文件位置：
- ppo_lift_final_v3.zip          (60 万步，旧版)
- ppo_lift_final_v4_0323.zip     (v2 训练，可能不稳定)
- ppo_lift_final_v5_0323.zip     (v3 训练，稳定版)
- ppo_lift_final_v6_0323_continued_*.zip  (继续训练版，最新)
"""

from stable_baselines3 import PPO
import robosuite as suite
import numpy as np
import time
import os

print("=" * 70)
print("使用已训练好的 PPO 模型")
print("=" * 70)

# =============================================================================
# 第 1 步：选择要加载的模型
# =============================================================================

print("""
可用的模型文件：
1. ppo_lift_final_v3.zip           - 旧版（60 万步）
2. ppo_lift_final_v4_0323.zip      - v2 训练（可能不稳定）
3. ppo_lift_final_v5_0323.zip      - v3 稳定版
4. ppo_lift_final_v6_0323_continued_*.zip - 继续训练版（推荐）
""")

# 自动找到最新的 v6 模型
model_files = [f for f in os.listdir('.') if f.startswith('ppo_lift_final_v6') and f.endswith('.zip')]
if model_files:
    model_files.sort()
    model_path = model_files[-1]  # 选择最新的
    print(f"✅ 自动选择最新模型：{model_path}")
else:
    # 如果没有 v6 模型，使用 v5
    model_path = "ppo_lift_final_v5_0323.zip"
    print(f"⚠️  未找到 v6 模型，使用 v3 稳定版：{model_path}")

# =============================================================================
# 第 2 步：加载模型
# =============================================================================

print("\n加载模型中...")

try:
    model = PPO.load(model_path)
    print("✅ 模型加载成功！")
except FileNotFoundError:
    print(f"❌ 找不到模型文件：{model_path}")
    print("请先运行训练脚本生成模型！")
    exit()

# =============================================================================
# 第 3 步：创建环境
# =============================================================================

print("\n创建环境...")
print("会弹出一个窗口显示机器人和方块")

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,           # 开启渲染窗口
    has_offscreen_renderer=True,
    use_camera_obs=False,
    use_object_obs=True,
    reward_shaping=False,        # 使用自定义奖励（与训练时一致）
    control_freq=20,
    horizon=500,
)

print("✅ 环境创建成功！")

# =============================================================================
# 辅助函数：把字典观测值转成向量
# =============================================================================

def obs_to_vector(obs):
    """把字典格式的观测值拼成一个向量"""
    return np.concatenate([
        np.array(obs[k]).flatten() 
        for k in obs.keys() 
        if isinstance(obs[k], np.ndarray)
    ]).astype(np.float32)

# =============================================================================
# 第 4 步：运行测试
# =============================================================================

print("\n" + "=" * 70)
print("开始测试")
print("=" * 70)
print("观察机器人是否能成功举起方块！")
print("按 Ctrl+C 可以随时停止")

total_reward = 0
success_count = 0
num_episodes = 10  # 测试 10 次
TABLE_HEIGHT = 0.83  # 桌面高度
LIFT_THRESHOLD = 0.04  # 成功标准

for episode in range(num_episodes):
    obs_dict = env.reset()
    obs = obs_to_vector(obs_dict)
    episode_reward = 0
    max_height = 0
    actual_success = False  # 真正的成功标志
    
    print(f"\n--- 第 {episode+1}/{num_episodes} 次尝试 ---")
    
    for step in range(500):
        # 使用模型预测动作
        action, _ = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs_dict, reward, done, info = env.step(action)
        obs = obs_to_vector(obs_dict)
        episode_reward += reward
        
        # 追踪最大高度
        if 'cube_pos' in obs_dict:
            height = obs_dict['cube_pos'][2]
            max_height = max(max_height, height)
        
        # 检查真正的成功条件（使用环境的内部方法）
        try:
            if hasattr(env, '_check_success'):
                actual_success = env._check_success()
        except:
            pass
        
        # 渲染显示
        env.render()
        time.sleep(0.02)  # 放慢速度方便观察
        
        # 打印夹爪状态
        if step % 50 == 0:
            gripper_qpos = obs_dict.get('robot0_gripper_qpos', None)
            if gripper_qpos is not None:
                print(f"  步{step}: 夹爪位置={gripper_qpos}, 高度={height:.4f}米")
        
        if done:
            # 不再根据 done 判断成功，而是根据实际高度
            relative_height = max_height - TABLE_HEIGHT
            if relative_height > LIFT_THRESHOLD and actual_success:
                print(f"✅ 真正成功！步数：{step}, 奖励：{episode_reward:.2f}, 高度：{relative_height:.4f}米")
                success_count += 1
            else:
                print(f"❌ 未成功！步数：{step}, 奖励：{episode_reward:.2f}, 高度：{relative_height:.4f}米")
            break
    
    # 打印本次尝试的最大高度
    relative_height = max_height - TABLE_HEIGHT
    print(f"最大相对高度：{relative_height:.4f}米 (成功标准：>{LIFT_THRESHOLD}米)")
    if not actual_success:
        print("⚠️  环境内部成功检查：失败")
    
    total_reward += episode_reward

# =============================================================================
# 第 5 步：打印统计
# =============================================================================

print("\n" + "=" * 70)
print("测试完成！统计结果：")
print("=" * 70)
print(f"测试次数：{num_episodes} 次")
print(f"成功次数：{success_count} 次")
print(f"成功率：{success_count/num_episodes*100:.1f}%")
print(f"平均奖励：{total_reward/num_episodes:.2f}")

# 评价
if success_count >= 8:
    print("\n🎉 模型表现优秀！成功率 >= 80%")
elif success_count >= 5:
    print("\n👍 模型表现良好！成功率 >= 50%")
else:
    print("\n📈 模型还有提升空间，建议继续训练")

env.close()
print("\n测试结束！窗口已关闭。")