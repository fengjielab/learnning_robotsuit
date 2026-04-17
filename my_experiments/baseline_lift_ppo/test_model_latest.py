"""
===============================================================================
                    测试训练好的 PPO 模型（修复版）
===============================================================================

这个脚本会：
1. 加载训练好的模型
2. 创建带渲染的环境
3. 让模型控制机器人完成任务
4. 显示成功率统计
"""

from stable_baselines3 import PPO
import robosuite as suite
import numpy as np
import time

print("=" * 70)
print("测试训练好的 PPO 模型")
print("=" * 70)

# =============================================================================
# 第 1 步：加载模型
# =============================================================================

model_path = "ppo_lift_final_v3.zip"  # 使用最新模型（约 60 万步）
print(f"加载模型：{model_path}")
print(f"这是最新模型，训练了约 60 万步！")

try:
    model = PPO.load(model_path)
    print("✅ 模型加载成功！")
except FileNotFoundError:
    print(f"❌ 找不到模型文件：{model_path}")
    print("请先运行训练脚本生成模型！")
    exit()

# =============================================================================
# 第 2 步：创建环境
# =============================================================================

print("\n创建环境...")
print("会弹出一个窗口显示机器人和方块")

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=True,
    use_camera_obs=False,
    use_object_obs=True,
    reward_shaping=True,
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
# 第 3 步：运行测试
# =============================================================================

print("\n开始测试（按 Ctrl+C 停止）...")
print("观察机器人是否能成功举起方块！")

total_reward = 0
success_count = 0
num_episodes = 10  # 测试 10 次

for episode in range(num_episodes):
    obs_dict = env.reset()
    obs = obs_to_vector(obs_dict)  # 转成向量
    episode_reward = 0
    
    print(f"\n--- 第 {episode+1}/{num_episodes} 次尝试 ---")
    
    for step in range(500):
        # 使用模型预测动作
        action, _ = model.predict(obs, deterministic=True)
        
        # 执行动作
        obs_dict, reward, done, info = env.step(action)
        obs = obs_to_vector(obs_dict)  # 转成向量
        episode_reward += reward
        
        # 渲染显示
        env.render()
        time.sleep(0.02)  # 放慢速度方便观察（0.02 秒 = 50FPS）
        
        if done:
            print(f"✅ 成功！步数：{step}, 奖励：{episode_reward:.2f}")
            success_count += 1
            break
    
    total_reward += episode_reward

# =============================================================================
# 第 4 步：打印统计
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