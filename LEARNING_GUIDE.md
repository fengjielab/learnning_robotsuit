# robosuite 学习指南

## 📚 项目概览

**robosuite** 是一个基于 MuJoCo 物理引擎的机器人仿真框架，专为机器人学习研究设计。

### 核心特性
- **标准化任务**: 9 个基准测试环境（Lift, Stack, Door, PickPlace 等）
- **模块化设计**: 可组合机器人、夹爪、场景、物体创建新环境
- **多种控制器**: 关节控制、操作空间控制、逆运动学等
- **多模态传感器**: RGB 相机、深度图、本体感知
- **人机交互**: 支持键盘/空间鼠标遥操作

---

## 🗺️ 学习路线图

### 阶段 1：环境探索（你已完成 ✅）
你已经创建了很好的探索脚本：

| 脚本 | 目的 |
|------|------|
| `01_explore_structure.py` | 了解包结构和可用资源 |
| `02_env_showcase.py` | 体验不同任务环境 |
| `03_robot_comparision.py` | 对比 6 种机器人外观 |
| `04_controller_feel.py` | 感受不同控制器风格 |

### 阶段 2：深入理解（下一步）

#### 2.1 阅读官方文档
```
docs/
├── quickstart.md      # 快速入门（已读过）
├── modules/
│   ├── environments.md  # 环境 API
│   ├── robots.md        # 机器人模型
│   ├── controllers.md   # 控制器详解
│   └── sensors.md       # 传感器系统
└── modeling/          # 如何构建自定义环境
```

#### 2.2 运行官方 Demo
```bash
cd robosuite/robosuite/demos/

# 基础控制演示
python demo_control.py

# 随机动作演示
python demo_random_action.py

# 设备控制（需要 SpaceMouse）
python demo_device_control.py

# 视频录制
python demo_video_recording.py
```

#### 2.3 理解核心概念

**环境架构**:
```
Task (任务)
├── Robot Model (机器人)
│   └── Gripper (夹爪)
├── Arena (场景/工作台)
└── Object Models (物体)
```

**控制层级**:
```
用户/策略 → 控制器 → 关节扭矩 → MuJoCo 物理引擎 → 状态更新
                ↓
            传感器 → 观测值
```

### 阶段 3：实践项目

#### 项目 A：自定义环境
创建一个简单的"推箱子"任务：
```python
from robosuite.models import MujocoWorldBase
from robosuite.models.robots import Panda
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject

# 1. 创建世界
world = MujocoWorldBase()

# 2. 添加机器人
robot = Panda()
robot.add_gripper(...)
world.merge(robot)

# 3. 添加桌子
arena = TableArena()
world.merge(arena)

# 4. 添加箱子
box = BoxObject(size=[0.05, 0.05, 0.05])
world.merge(box)

# 5. 加载仿真
model = world.get_model(mode="mujoco")
```

#### 项目 B：模仿学习
使用 `robomimic` 项目学习从演示中模仿：
```bash
# 收集演示数据
python robosuite/scripts/collect_human_demonstrations.py

# 使用 robomimic 训练策略
# https://github.com/ARISE-Initiative/robomimic
```

#### 项目 C：强化学习
使用 Stable Baselines3 训练 RL 策略：
```python
import robosuite as suite
from stable_baselines3 import PPO

env = suite.make("Lift", has_renderer=False, ...)
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
```

---

## 📖 推荐学习顺序

### 第 1 周：基础熟悉
1. ✅ 运行你的 4 个实验脚本
2. 阅读 `docs/quickstart.md` 和 `docs/modules/overview.md`
3. 运行 `demos/demo_control.py` 理解控制器行为

### 第 2 周：深入 API
1. 阅读 `docs/modeling/` 下的文档
2. 查看 `robosuite/environments/manipulation/` 源码
3. 尝试修改现有环境的参数

### 第 3 周：动手实践
1. 创建一个简单的自定义物体
2. 修改 Lift 环境，添加多个物体
3. 尝试收集演示数据

### 第 4 周：进阶应用
1. 集成 RL 库（SB3 / RLlib）
2. 或尝试模仿学习（robomimic）
3. 设计自己的基准任务

---

## 🔧 常用代码片段

### 创建环境
```python
import robosuite as suite

env = suite.make(
    env_name="Lift",
    robots="Panda",
    controller_configs=suite.load_controller_config(default_controller="OSC_POSE"),
    has_renderer=True,          # 显示窗口
    has_offscreen_renderer=False,
    use_camera_obs=False,       # 不使用相机观测
    control_freq=20,            # 控制频率
    horizon=1000,               # 每集步数
    ignore_done=False,          # 任务完成后是否结束
)
```

### 运行循环
```python
env.reset()
for i in range(1000):
    action = np.random.randn(env.action_dim)
    obs, reward, done, info = env.step(action)
    env.render()
    if done:
        break
env.close()
```

### 观测值结构
```python
# obs 是 OrderedDict，包含：
obs.keys()
# dict_keys(['robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos', ...])

# 访问特定观测
eef_position = obs['robot0_eef_pos']
```

---

## 📁 目录结构说明

```
robosuite/
├── robosuite/              # 核心代码
│   ├── environments/       # 环境定义
│   │   └── manipulation/   # 操作任务（Lift, Stack, Door...）
│   ├── models/             # MJCF 模型构建器
│   │   ├── robots/         # 机器人模型
│   │   ├── grippers/       # 夹爪模型
│   │   ├── objects/        # 物体模型
│   │   └── arenas/         # 场景模型
│   ├── controllers/        # 控制器实现
│   ├── robots/             # 机器人包装类
│   ├── devices/            # 输入设备（键盘、SpaceMouse）
│   ├── sensors/            # 传感器
│   ├── renderers/          # 渲染器
│   ├── wrappers/           # Gym 等包装器
│   └── demos/              # 演示脚本
├── docs/                   # 文档
├── my_experiments/         # 你的实验
└── notebooks/              # Jupyter 笔记本
```

---

## 🌐 外部资源

- **官方网站**: https://robosuite.ai/
- **论文**: https://arxiv.org/abs/2009.12293
- **GitHub**: https://github.com/ARISE-Initiative/robosuite
- **姊妹项目**:
  - [robomimic](https://github.com/ARISE-Initiative/robomimic) - 模仿学习
  - [RoboTurk](https://github.com/robosuite/robosuite/tree/master/robosuite/demos) - 人类演示数据集

---

## 💡 学习建议

1. **多看源码**: `robosuite/environments/manipulation/` 下的环境实现是最好的教程
2. **从小改起**: 先修改现有环境的参数，再尝试组合模块，最后从头构建
3. **善用可视化**: 使用 `has_renderer=True` 实时查看效果
4. **理解 MJCF**: 学习 MuJoCo 的 XML 格式有助于深入理解模型
5. **加入社区**: GitHub Issues 和 Discussions 有很多有价值的讨论

---

## 📝 下一步行动清单

- [ ] 运行所有官方 demo 脚本
- [ ] 阅读 `docs/modules/controllers.md`
- [ ] 查看 `robosuite/environments/manipulation/lift.py` 源码
- [ ] 尝试修改 Lift 环境的物体大小/颜色
- [ ] 创建一个包含多个物体的自定义环境

祝你学习顺利！🚀