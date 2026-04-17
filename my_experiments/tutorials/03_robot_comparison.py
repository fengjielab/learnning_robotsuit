import robosuite as suite
import numpy as np

# 6种机器人清单
robots = [
    ("IIWA", "KUKA 工业臂，橙色，6轴，精度高"),
    ("Jaco", "Kinova 轻量臂，黑色，7轴，像章鱼触手"),
    ("Kinova3", "Kinova Gen3，银灰，7轴，带视觉模块"),
    ("Panda", "Franka Panda，白红，7轴，最流行"),
    ("Sawyer", "Rethink Sawyer，黑红，7轴，已停产"),
    ("UR5e", "Universal Robots，蓝/灰，6轴，工业标准"),
]

print("🤖 机器人外观对比实验")
print("固定任务：Lift（举方块），每个机器人跑 100 步（5秒）")
print("观察重点：颜色、臂长、关节数量、夹爪形状、动作风格")
print("=" * 60)

for robot_name, description in robots:
    print(f"\n🎬 正在加载：{robot_name}")
    print(f"   特征：{description}")
    input("   按 Enter 开始（或 Ctrl+C 跳过）...")
    
    try:
        # 创建环境
        env = suite.make(
            env_name="Lift",
            robots=robot_name,
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
        )
        
        env.reset()
        
        # 获取动作维度（修复点：不用 action_space）
        action_dim = env.action_dim
        print(f"   动作维度：{action_dim}（关节数+夹爪）")
        
        # 运行 100 步
        for i in range(100):
            # 生成随机动作（修复点：用 action_dim）
            action = np.random.randn(action_dim) * 0.3  # 幅度小一点，别太抽搐
            obs, reward, done, info = env.step(action)
            env.render()
            
        env.close()
        print(f"   ✅ {robot_name} 展示完成")
        
    except KeyboardInterrupt:
        print(f"   ⏭️  跳过 {robot_name}")
        continue
    except Exception as e:
        print(f"   ❌ 报错：{str(e)[:50]}...")  # 只显示前50字符
        continue

print("\n" + "=" * 60)
print("📝 观察总结指南：")
print("1. 颜色：Panda(白红) | UR5e(蓝/灰) | IIWA(橙) | Jaco(黑)")
print("2. 轴数：6轴(IIWA, UR5e) 动作刚硬 | 7轴(其他) 更灵活")
print("3. 臂长：UR5e 最长 | Jaco 最短最细")
print("4. 夹爪：Panda(平行夹爪) | Jaco(三指爪) | UR5e(可选多种)")