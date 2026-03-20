"""
阶段 5：深入学习 - 探索 robosuite 的核心 API

这个脚本帮助你：
1. 查看所有可用的环境详情
2. 理解观测值的结构
3. 探索不同控制器的行为
4. 查看如何自定义环境
"""

import robosuite as suite
import numpy as np

print("=" * 70)
print("🎓 robosuite 深入学习指南")
print("=" * 70)

# ==================== 1. 环境清单 ====================
print("\n📋 1. 所有可用环境")
print("-" * 50)

# 查看环境中模块获取所有环境
from robosuite.environments.manipulation import *

env_classes = [
    ("Lift", "举起方块到指定高度"),
    ("Stack", "将一个物体堆到另一个上面"),
    ("NutAssembly", "将螺母拧到螺栓上"),
    ("PickPlace", "分拣物体到指定箱子"),
    ("Door", "打开门/抽屉"),
    ("Wipe", "擦拭桌面"),
    ("CleanUp", "清理桌面物体"),
    ("TwoArmPegInHole", "双臂 peg-in-hole 装配"),
]

for name, desc in env_classes:
    print(f"  • {name:20s} - {desc}")

# ==================== 2. 机器人清单 ====================
print("\n🤖 2. 所有可用机器人")
print("-" * 50)

robots_info = [
    ("Panda", "Franka Emika", "7 轴，最流行，研究首选"),
    ("Sawyer", "Rethink Robotics", "7 轴，已停产但好用"),
    ("IIWA", "KUKA", "6 轴，橙色工业臂"),
    ("UR5e", "Universal Robots", "6 轴，工业标准"),
    ("Jaco", "Kinova", "7 轴，轻量级"),
    ("Kinova3", "Kinova Gen3", "7 轴，现代化设计"),
    ("Baxter", "Rethink Robotics", "双臂，研究平台"),
]

print(f"{'名称':<12} {'厂商':<20} {'特点':<20}")
print("-" * 52)
for name, vendor, feature in robots_info:
    print(f"{name:<12} {vendor:<20} {feature:<20}")

# ==================== 3. 控制器详解 ====================
print("\n🎮 3. 控制器类型详解")
print("-" * 50)

controllers = [
    ("JOINT_POSITION", "关节位置", "直接控制每个关节角度"),
    ("JOINT_VELOCITY", "关节速度", "控制关节转动速度"),
    ("JOINT_TORQUE", "关节扭矩", "直接控制电机扭矩"),
    ("OSC_POSITION", "操作空间位置", "控制末端 3D 位置"),
    ("OSC_POSE", "操作空间位姿", "控制末端位置 + 朝向"),
    ("IK_POSE", "逆运动学位姿", "类似 OSC 但计算方式不同"),
]

print(f"{'控制器':<20} {'简称':<15} {'说明':<25}")
print("-" * 60)
for ctrl, short, desc in controllers:
    print(f"{ctrl:<20} {short:<15} {desc:<25}")

# ==================== 4. 观测值结构探索 ====================
print("\n👁️  4. 观测值结构示例")
print("-" * 50)

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
)

env.reset()
obs, _, _, _ = env.step(np.zeros(env.action_dim))

print(f"观测值类型：{type(obs)}")
print(f"观测值数量：{len(obs)} 项")
print("\n观测值详情:")
for key, value in obs.items():
    if hasattr(value, 'shape'):
        print(f"  • {key:30s} 形状：{value.shape}")
    else:
        print(f"  • {key:30s} 值：{value}")

env.close()

# ==================== 5. 奖励函数探索 ====================
print("\n🏆 5. 奖励函数示例")
print("-" * 50)

env = suite.make(
    env_name="Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    reward_shaping=True,  # 启用奖励塑形
)

env.reset()
print("运行 10 步，观察奖励变化:")
for i in range(10):
    action = np.random.randn(env.action_dim) * 0.5
    obs, reward, done, info = env.step(action)
    print(f"  步{i+1}: 奖励={reward:.4f}, 完成={done}")

env.close()

# ==================== 6. 自定义环境入门 ====================
print("\n🔧 6. 自定义环境入门")
print("-" * 50)

print("""
要创建自定义环境，你需要了解以下模块：

from robosuite.models import MujocoWorldBase      # 世界基类
from robosuite.models.robots import Panda         # 机器人模型
from robosuite.models.grippers import PandaGripper # 夹爪模型
from robosuite.models.arenas import TableArena    # 桌子场景
from robosuite.models.objects import BoxObject    # 物体模型

基本步骤:
1. 创建 MujocoWorldBase 实例
2. 添加机器人（带夹爪）
3. 添加场景（桌子、地板等）
4. 添加物体（箱子、球等）
5. 调用 get_model(mode="mujoco") 获取仿真模型
6. 使用 mujoco 库运行仿真

示例代码位置:
- docs/quickstart.md - 快速入门示例
- robosuite/environments/manipulation/lift.py - Lift 环境源码
- robosuite/demos/ - 各种功能演示
""")

# ==================== 7. 学习资源总结 ====================
print("\n📚 7. 学习资源总结")
print("-" * 50)

print("""
文档阅读顺序:
1. docs/quickstart.md        - 快速开始
2. docs/modules/overview.md  - 架构概览
3. docs/modules/environments.md - 环境 API
4. docs/modules/controllers.md  - 控制器详解
5. docs/modeling/            - 建模 API

源码阅读顺序:
1. robosuite/environments/__init__.py  - 环境注册
2. robosuite/environments/robot_env.py - 环境基类
3. robosuite/environments/manipulation/lift.py - 简单环境示例
4. robosuite/controllers/              - 控制器实现

实践建议:
1. 先运行所有 demos/ 下的脚本
2. 修改现有环境参数（物体大小、颜色等）
3. 组合不同机器人和环境
4. 尝试创建自定义物体
5. 集成 RL 库进行训练
""")

print("=" * 70)
print("✅ 探索完成！查看 LEARNING_GUIDE.md 获取详细学习路线")
print("=" * 70)