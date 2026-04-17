"""
===============================================================================
检查 Lift 任务的成功标准和当前模型表现（修复版 - 使用正确的坐标系）
===============================================================================

重要说明：
- robosuite 的桌面高度是 0.83 米（世界坐标系）
- 方块初始位置在桌面上，所以 Z = 0.83 米
- 成功标准：方块被举起超过 4 厘米 = Z > 0.87 米
- 相对高度 = 方块 Z - 0.83 米
"""

from stable_baselines3 import PPO
import robosuite as suite
import numpy as np

# 常量
TABLE_HEIGHT = 0.83  # 桌面高度（米）
LIFT_THRESHOLD = 0.04  # 需要举起的高度（米）
SUCCESS_HEIGHT = TABLE_HEIGHT + LIFT_THRESHOLD  # 成功高度 = 0.87 米

print("=" * 70)
print("Lift 任务成功标准分析（修复版）")
print("=" * 70)
print(f"""
坐标系说明：
- 桌面高度：{TABLE_HEIGHT} 米
- 成功标准：方块 Z 坐标 > {SUCCESS_HEIGHT} 米（即举起 {LIFT_THRESHOLD} 米）
- 相对高度 = 方块 Z - {TABLE_HEIGHT} 米
""")

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

def get_cube_height(obs_dict):
    """从观测值获取方块高度（世界坐标系）"""
    if 'cube_pos' in obs_dict:
        return float(obs_dict['cube_pos'][2])
    return None

def get_relative_height(cube_height):
    """计算相对桌面的高度"""
    return cube_height - TABLE_HEIGHT

# =============================================================================
# 第 3 步：运行测试并记录详细数据
# =============================================================================

print("\n" + "=" * 70)
print("开始测试（按 Ctrl+C 停止）")
print("=" * 70)

all_max_heights = []
all_max_relative_heights = []
success_count = 0
num_episodes = 10

for episode in range(num_episodes):
    obs_dict = env.reset()
    obs = obs_to_vector(obs_dict)
    
    # 获取初始高度
    initial_height = get_cube_height(obs_dict)
    
    print(f"\n{'='*50}")
    print(f"第 {episode+1}/{num_episodes} 次尝试")
    print(f"{'='*50}")
    print(f"  初始高度：{initial_height:.4f}米（相对：{get_relative_height(initial_height):.4f}米）")
    
    max_height = initial_height
    max_relative_height = get_relative_height(initial_height)
    success = False
    
    for step in range(500):
        action, _ = model.predict(obs, deterministic=True)
        obs_dict, reward, done, info = env.step(action)
        obs = obs_to_vector(obs_dict)
        
        # 获取方块高度
        cube_height = get_cube_height(obs_dict)
        
        if cube_height is not None:
            relative_height = get_relative_height(cube_height)
            max_height = max(max_height, cube_height)
            max_relative_height = max(max_relative_height, relative_height)
            
            # 每 20 步显示一次高度
            if step % 20 == 0:
                status = "✅" if relative_height > LIFT_THRESHOLD else "  "
                print(f"  步{step:3d}: 高度 = {cube_height:.4f}米 (相对：{relative_height:.4f}米) {status}")
        
        env.render()
        
        # 检查成功 - 使用环境内部的成功标志
        try:
            if hasattr(env, '_check_success'):
                success = env._check_success()
            elif hasattr(env, 'success'):
                success = env.success
            else:
                #  fallback: 使用高度判断
                success = relative_height > LIFT_THRESHOLD
        except:
            success = False
        
        if success:
            print(f"\n  ✅ 任务成功！")
            success_count += 1
            break
    
    all_max_heights.append(max_height)
    all_max_relative_heights.append(max_relative_height)
    
    if not success:
        print(f"\n  ❌ 任务失败")
    
    print(f"  最大高度：{max_height:.4f}米")
    print(f"  最大相对高度：{max_relative_height:.4f}米")
    print(f"  成功标准：相对高度 > {LIFT_THRESHOLD} 米")
    if max_relative_height > LIFT_THRESHOLD:
        print(f"  ✅ 达到成功标准！")
    else:
        print(f"  ❌ 未达到标准，还差 {LIFT_THRESHOLD - max_relative_height:.4f}米")

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

高度统计（世界坐标系）:
  最小最大高度：{min(all_max_heights):.4f}米
  最大最大高度：{max(all_max_heights):.4f}米
  平均最大高度：{np.mean(all_max_heights):.4f}米

高度统计（相对桌面）:
  最小最大相对高度：{min(all_max_relative_heights):.4f}米
  最大最大相对高度：{max(all_max_relative_heights):.4f}米
  平均最大相对高度：{np.mean(all_max_relative_heights):.4f}米

成功标准：相对高度 > {LIFT_THRESHOLD} 米
""")

# =============================================================================
# 第 5 步：分析结果
# =============================================================================

print("=" * 70)
print("分析结果")
print("=" * 70)

avg_relative_height = np.mean(all_max_relative_heights)

if success_count >= 8:
    print("""
🎉 模型表现优秀！

模型已经学会了完整的 Lift 任务：
1. ✅ 接近方块
2. ✅ 抓取方块
3. ✅ 举起方块（超过 4 厘米）

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
    print(f"""
📈 模型还有提升空间！

平均最大相对高度：{avg_relative_height:.4f}米
成功标准：{LIFT_THRESHOLD} 米

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