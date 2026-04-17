"""
===============================================================================
检查 Lift 任务的成功标准和当前模型表现
===============================================================================

这个脚本会显示：
1. 每一步的方块高度
2. 最大高度
3. 是否达到成功标准（0.04 米）
4. 分析模型学到了什么
"""

from stable_baselines3 import PPO
import robosuite as suite
import numpy as np

print("=" * 70)
print("Lift 任务成功标准分析")
print("=" * 70)

# =============================================================================
# 第 1 步：加载模型
# =============================================================================

model_path = "ppo_lift_final_v3.zip"
print(f"加载模型：{model_path}")
model = PPO.load(model_path)
print("✅ 模型加载成功！")

# =============================================================================
# 第 2 步：创建环境
# =============================================================================

print("\n创建环境...")

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
# 辅助函数
# =============================================================================

def obs_to_vector(obs):
    """把字典格式的观测值拼成一个向量"""
    return np.concatenate([
        np.array(obs[k]).flatten() 
        for k in obs.keys() 
        if isinstance(obs[k], np.ndarray)
    ]).astype(np.float32)

# =============================================================================
# 第 3 步：运行测试并记录详细数据
# =============================================================================

print("\n" + "=" * 70)
print("开始测试（按 Ctrl+C 停止）")
print("=" * 70)
print("""
robosuite Lift 任务成功标准：
- 方块必须被举起超过 0.04 米（4 厘米）

显示说明：
- 每 20 步显示一次方块高度
- ✅ 表示超过成功标准
""")

all_max_heights = []
success_count = 0
num_episodes = 10

for episode in range(num_episodes):
    obs_dict = env.reset()
    obs = obs_to_vector(obs_dict)
    
    print(f"\n{'='*50}")
    print(f"第 {episode+1}/{num_episodes} 次尝试")
    print(f"{'='*50}")
    
    max_height = 0
    success = False
    heights = []
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs_dict, reward, done, info = env.step(action)
        obs = obs_to_vector(obs_dict)
        
        # 获取方块位置 - 修复：使用正确的键名
        # robosuite 返回的观测值中，物体位置可能是 "object-state" 或其他键
        height = None
        
        # 尝试多种方式获取方块高度
        for key in obs_dict.keys():
            if 'object' in key.lower() or 'cube' in key.lower():
                val = obs_dict[key]
                if isinstance(val, np.ndarray) and len(val) >= 3:
                    # 假设第 3 个值是 Z 坐标（高度）
                    h = float(val[2])
                    if height is None or h > height:
                        height = h
        
        # 如果还是没找到，尝试从环境直接获取
        if height is None and hasattr(env, 'sim'):
            try:
                if hasattr(env.model, 'cube_id'):
                    cube_pos = env.sim.data.body_xpos[env.model.cube_id]
                    height = float(cube_pos[2])
            except:
                pass
        
        if height is not None:
            max_height = max(max_height, height)
            
            # 每 20 步显示一次高度
            if step % 20 == 0:
                status = "✅" if height > 0.04 else "  "
                print(f"  步{step:3d}: 高度 = {height:.4f}米 {status}")
        
        env.render()
        
        # 检查成功 - 使用环境内部的成功标志，而不是 done
        try:
            if hasattr(env, '_check_success'):
                success = env._check_success()
            elif hasattr(env, 'success'):
                success = env.success
            else:
                success = done
        except:
            success = False
        
        if success:
            print(f"\n  ✅ 任务成功！")
            success_count += 1
            break
    
    all_max_heights.append(max_height)
    
    if not success:
        print(f"\n  ❌ 任务失败")
    
    print(f"  最大高度：{max_height:.4f}米")
    print(f"  成功标准：0.0400 米")
    if max_height > 0.04:
        print(f"  ✅ 达到成功标准！")
    else:
        print(f"  ❌ 未达到标准，还差 {0.04 - max_height:.4f}米")

# =============================================================================
# 第 4 步：统计分析
# =============================================================================

print("\n" + "=" * 70)
print("统计分析")
print("=" * 70)

print(f"""
测试次数：{num_episodes} 次
成功次数：{success_count} 次
成功率：{success_count/num_episodes*100:.1f}%

高度统计:
  最小最大高度：{min(all_max_heights):.4f}米
  最大最大高度：{max(all_max_heights):.4f}米
  平均最大高度：{np.mean(all_max_heights):.4f}米
  中位数高度：{np.median(all_max_heights):.4f}米

成功标准：0.0400 米
""")

# 分析
print("=" * 70)
print("分析结果")
print("=" * 70)

avg_height = np.mean(all_max_heights)

if success_count >= 8:
    print("""
🎉 模型表现优秀！

模型已经学会了完整的 Lift 任务：
1. ✅ 接近方块
2. ✅ 抓取方块
3. ✅ 举起方块

成功率 >= 80%，可以投入使用。
""")
elif success_count >= 5:
    print("""
👍 模型表现良好！

模型基本学会了 Lift 任务，但还不够稳定：
1. ✅ 接近方块
2. ✅ 抓取方块
3. ⚠️ 举起方块（有时成功）

建议：
- 继续训练更多步数（100 万步+）
- 或调整奖励函数
""")
else:
    print("""
📈 模型还有提升空间！

模型可能只学会了部分任务：
1. ✅ 接近方块
2. ⚠️ 抓取方块（可能没学会）
3. ❌ 举起方块（没学会）

问题分析：
- 模型可能陷入"局部最优"
- 只学会了接近方块，没学会后续动作

建议：
1. 增加训练步数到 100 万步
2. 调整奖励函数，增加"举起"的奖励权重
3. 使用课程学习，分阶段训练
""")

env.close()
print("\n测试结束！")