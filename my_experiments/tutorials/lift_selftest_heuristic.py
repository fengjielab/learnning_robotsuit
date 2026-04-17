
import robosuite as suite
import numpy as np

# =============================================================================
# 方法 1：简单的启发式策略（教学用）
# =============================================================================
"""
这不是真正的学习，而是手动编写规则让机器人完成任务。
目的：让你理解"策略"是什么，以及看到"好"的行为是什么样的。
"""

print("=" * 70)
print("方法 1：启发式策略 - 手动编写规则")
print("=" * 70)

def heuristic_policy(env, obs):
    """
    简单的启发式策略：
    1. 如果没抓住方块 → 向方块移动
    2. 如果抓住了方块 → 向上移动
    """
    # 获取观测值
    gripper_pos = obs['robot0_eef_pos']  # 夹爪位置
    cube_pos = obs['cube_pos']           # 方块位置
    
    # 计算方向
    to_cube = cube_pos - gripper_pos     # 到方块的方向
    
    # 检查是否抓住（需要从观测值中获取或估计）
    # 这里简化：如果夹爪接近方块，假设抓住了
    distance = np.linalg.norm(to_cube)
    
    if distance < 0.05:  # 5 厘米以内，假设抓住了
        # 向上移动
        action = np.array([0, 0, 0.1, 0, 0, 0, 0, 0.5])  # 向上 + 闭合夹爪
    else:
        # 向方块移动
        action = np.array([to_cube[0], to_cube[1], to_cube[2], 0, 0, 0, 0, -0.5])  # 移动 + 张开夹爪
    
    # 限制动作幅度
    action = np.clip(action, -1, 1)
    
    return action

# 测试启发式策略
def test_heuristic():
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=True,
        reward_shaping=True,
        control_freq=20,
        horizon=500,
    )
    
    obs = env.reset()
    total_reward = 0
    
    for i in range(500):
        # 使用启发式策略而不是随机动作
        action = heuristic_policy(env, obs)
        
        obs, reward, done, info = env.step(action)
        env.render()
        total_reward += reward
        
        if i % 50 == 0:
            print(f"步{i}: 奖励={reward:.4f}, 总奖励={total_reward:.4f}")
        
        if done:
            break
    
    print(f"\n启发式策略总奖励：{total_reward:.4f}")
    env.close()
test_heuristic()