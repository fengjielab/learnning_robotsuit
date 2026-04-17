import robosuite as suite
from robosuite.controllers import load_controller_config
import numpy as np

controllers = [
    ("JOINT_POSITION", "关节位置控制：直接扭关节"),
    ("OSC_POSITION", "操作空间位置：控制手往哪移动"),
    ("OSC_POSE", "操作空间位姿：控制手的位置+朝向"),
]

print("🎮 控制器手感对比（修复版）")
print("观察：动作是'关节扭曲'还是'整体平移'？")
print("=" * 60)

for ctrl_name, description in controllers:
    print(f"\n🎮 测试：{ctrl_name}")
    print(f"   说明：{description}")
    input("   按 Enter 开始...")
    
    try:
        # 加载控制器配置
        config = load_controller_config(default_controller=ctrl_name)
        
        env = suite.make(
            env_name="Lift",
            robots="Panda",
            controller_configs=config,
            has_renderer=True,
            control_freq=20,
        )
        
        env.reset()
        
        # 关键修复：用 env.action_dim 获取维度，不硬猜！
        dim = env.action_dim
        print(f"   动作维度：{dim}")
        
        for i in range(150):
            # 根据不同控制器生成不同动作
            if "JOINT" in ctrl_name:
                # 关节控制：尝试伸展手臂（大致向上）
                # 前7个是关节，最后1个是夹爪
                action = np.array([0, -0.5, 0, -1.5, 0, 1.5, 0, 0.5])
            elif ctrl_name == "OSC_POSITION":
                # OSC位置：只有4维 [dx, dy, dz, gripper]
                # 向上移动
                action = np.array([0, 0, 0.02, 0.5])
            else:  # OSC_POSE
                # OSC位姿：7维 [dx, dy, dz, roll, pitch, yaw, gripper]
                action = np.array([0, 0, 0.02, 0, 0, 0, 0.5])
            
            # 加一点随机扰动
            noise = np.random.randn(dim) * 0.05
            action = action[:dim] + noise  # 确保维度匹配
            
            obs, reward, done, info = env.step(action)
            env.render()
            
        env.close()
        print(f"   ✅ 完成")
        
    except Exception as e:
        print(f"   ❌ 错误：{e}")

print("\n💡 观察要点：")
print("- JOINT：7个关节各自扭动，像木偶")
print("- OSC_POSITION：末端直线移动，像被无形的手拖动")
print("- OSC_POSE：还能转手腕，可以'竖着'抓取")