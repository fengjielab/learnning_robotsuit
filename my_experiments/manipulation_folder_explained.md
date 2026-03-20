# manipulation 文件夹详解

## 📌 一句话总结

**`manipulation` 文件夹包含了 robosuite 中所有"操作任务"的环境定义**——就是那些让机器人用手臂抓取、移动、装配物体的任务。

---

## 🗂️ 文件夹内容

```
manipulation/
├── manipulation_env.py    ← 基类：所有操作环境的父类
├── single_arm_env.py      ← 单臂环境基类
├── two_arm_env.py         ← 双臂环境基类
│
├── lift.py                ← 举起方块
├── stack.py               ← 堆叠方块
├── can.py                 ← 分拣罐头
├── nut_assembly.py        ← 螺母装配
├── door.py                ← 开门
├── wipe.py                ← 擦桌子
├── pick_place.py          ← 抓取放置（多物体）
├── tool_hang.py           ← 工具悬挂
│
└── two_arm_handover.py    ← 双臂传递
    two_arm_lift.py        ← 双臂举物
    two_arm_peg_in_hole.py ← 双臂 peg-in-hole
    two_arm_transport.py   ← 双臂运输
```

---

## 🎯 这个文件夹是干啥的？

### 核心作用

`manipulation` 文件夹定义了 **9 个标准操作任务 + 4 个双臂任务**，每个任务都是一个完整的机器人学习环境。

### 什么是"操作任务"？

**操作（Manipulation）** = 机器人用手臂和夹爪与物体交互

| 任务 | 操作内容 |
|------|----------|
| Lift | 抓起一个方块并举起 |
| Stack | 把一个方块堆到另一个上面 |
| Can | 把罐头从箱子抓到对应颜色的桶里 |
| NutAssembly | 把螺母拧到对应的桩上 |
| Door | 转动门把手并开门 |
| Wipe | 用板擦擦白板 |
| PickPlace | 分拣多个物体到不同位置 |
| ToolHang | 把工具挂到挂钩上 |

---

## 🏗️ 类继承结构

```
                    RobotEnv (机器人环境基类)
                         ↑
                         │
              ManipulationEnv (操作环境基类)
                    /              \
                   /                \
      SingleArmEnv (单臂)        TwoArmEnv (双臂)
           ↑                          ↑
           │                          │
    ┌──────┼──────┐           ┌───────┼────────┐
    │      │      │           │       │        │
  Lift  Stack  Can...    TwoArmLift  TwoArmHandover...
```

---

## 📄 核心文件详解

### 1. manipulation_env.py（基类）

这是所有操作环境的**父类**，提供了操作任务通用的功能：

```python
class ManipulationEnv(RobotEnv):
    """
    操作环境的通用基类
    
    核心功能：
    1. 初始化机器人、夹爪、控制器
    2. 提供相机观测（RGB、深度、分割）
    3. 提供渲染功能
    """
```

#### 关键方法

| 方法 | 作用 |
|------|------|
| `_check_grasp()` | 检查夹爪是否抓住了物体 |
| `_gripper_to_target()` | 计算夹爪到目标的距离 |
| `_visualize_gripper_to_target()` | 用颜色可视化距离（红→绿） |
| `_check_robot_configuration()` | 验证机器人配置是否合法 |

#### 重要参数

```python
def __init__(
    self,
    robots,                    # 机器人类型： "Panda" 或 ["Panda", "Sawyer"]
    env_configuration="default",  # 环境配置
    controller_configs=None,   # 控制器配置
    gripper_types="default",   # 夹爪类型
    use_camera_obs=True,       # 是否使用相机观测
    has_renderer=False,        # 是否显示窗口
    control_freq=20,           # 控制频率 (Hz)
    horizon=1000,              # 每集步数
    camera_names="agentview",  # 相机名称
    camera_heights=256,        # 相机高度
    camera_widths=256,         # 相机宽度
):
```

---

### 2. single_arm_env.py（单臂基类）

单臂操作环境的基类，定义了单臂任务的通用配置。

```python
class SingleArmEnv(ManipulationEnv):
    """单臂操作环境的基类"""
    
    # 定义单臂环境的默认配置
    # 如机器人位置、桌子位置等
```

---

### 3. two_arm_env.py（双臂基类）

双臂操作环境的基类，处理双臂协调任务。

```python
class TwoArmEnv(ManipulationEnv):
    """双臂操作环境的基类"""
    
    # 支持多种双臂配置：
    # - "bimanual": 双臂相对
    # - "single-arm-parallel": 双臂平行
    # - "single-arm-opposed": 双臂相对
```

---

## 🎮 具体任务文件

### lift.py（举起方块）

```python
"""
任务：机器人抓起桌上的方块并举到指定高度

场景元素：
- 1 个机器人臂
- 1 个方块
- 1 张桌子

成功条件：方块高度 > 桌子高度 + 0.10 米
"""
```

### stack.py（堆叠方块）

```python
"""
任务：把一个方块堆到另一个方块上面

场景元素：
- 1 个机器人臂
- 2 个方块（不同颜色）
- 1 张桌子

成功条件：上方块高度 > 下方块高度 + 阈值
"""
```

### can.py（分拣罐头）

```python
"""
任务：把箱子中的罐头抓到对应颜色的桶里

场景元素：
- 1 个机器人臂
- 4 个罐头（不同颜色）
- 1 个箱子
- 4 个目标桶

成功条件：所有罐头在对应颜色的桶中
"""
```

### nut_assembly.py（螺母装配）

```python
"""
任务：把螺母拧到对应的桩上

场景元素：
- 1 个机器人臂
- 2 个桩（方形 + 圆形）
- 2 个螺母（方形 + 圆形）
- 1 张桌子

成功条件：方形螺母在方形桩上，圆形螺母在圆形桩上
"""
```

### door.py（开门）

```python
"""
任务：转动门把手并打开门

场景元素：
- 1 个机器人臂
- 1 扇门（带把手）

成功条件：门打开角度 > 阈值
"""
```

---

## 💻 使用示例

### 创建 Lift 环境

```python
import robosuite as suite

# 最简单的创建方式
env = suite.make(
    "Lift",              # 环境名称
    robots="Panda",      # 机器人
    has_renderer=True,   # 显示窗口
)

env.reset()
for i in range(100):
    action = env.action_spec.sample()  # 随机动作
    obs, reward, done, info = env.step(action)
    env.render()
env.close()
```

### 创建带相机的环境

```python
env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,        # 不显示窗口
    has_offscreen_renderer=True,  # 需要离屏渲染
    use_camera_obs=True,       # 使用相机观测
    camera_names=["agentview", "robot0_eye_in_hand"],
    camera_heights=84,
    camera_widths=84,
)
```

### 创建双臂环境

```python
env = suite.make(
    "TwoArmLift",             # 双臂举起任务
    robots=["Panda", "Sawyer"],  # 两个不同的机器人
    env_configuration="single-arm-opposed",  # 双臂相对
    has_renderer=True,
)
```

---

## 🔍 如何阅读这些文件

每个任务文件（如 `lift.py`）的结构：

```python
class Lift(SingleArmEnv):
    
    def __init__(self, ...):
        # 1. 调用父类初始化
        super().__init__(...)
        
        # 2. 设置任务特定参数
        self.object_initializer = ...
        
    def _load_model(self):
        # 3. 加载 MJCF 模型
        # - 创建场景（桌子、地板）
        # - 创建物体（方块、螺母等）
        # - 合并到世界中
        
    def _setup_references(self):
        # 4. 设置仿真引用
        # - 获取物体位置、关节位置的句柄
        
    def _setup_observables(self):
        # 5. 设置观测值
        # - 机器人本体感知
        # - 物体状态
        # - 相机图像
        
    def _check_success(self):
        # 6. 检查任务是否成功
        # - 返回 True/False
        
    def reward(self, action):
        # 7. 计算奖励
        # - 稀疏奖励或稠密奖励（奖励塑形）
        
    def visualization_seq(self):
        # 8. 可视化序列（可选）
```

---

## 📊 任务分类

### 按难度

| 难度 | 任务 |
|------|------|
| ⭐ | Lift（举起） |
| ⭐⭐ | Stack（堆叠） |
| ⭐⭐ | Door（开门） |
| ⭐⭐⭐ | NutAssembly（装配） |
| ⭐⭐⭐ | Can（分拣） |
| ⭐⭐⭐⭐ | PickPlace（多物体分拣） |
| ⭐⭐⭐⭐ | Wipe（擦拭） |

### 按技能类型

| 技能 | 任务 |
|------|------|
| 抓取 | Lift, Stack |
| 放置 | Stack, PickPlace, Can |
| 精细操作 | NutAssembly, ToolHang |
| 接触力控制 | Door, Wipe |
| 双臂协调 | TwoArmLift, TwoArmHandover, TwoArmPegInHole |

---

## 🎓 学习建议

### 阅读顺序

1. **先看 `lift.py`** - 最简单的任务，理解基本结构
2. **再看 `stack.py`** - 多一个物体，学习物体交互
3. **然后看 `manipulation_env.py`** - 理解基类功能
4. **最后看 `pick_place.py`** - 最复杂的单臂任务

### 实践建议

```bash
# 1. 运行每个环境看看
python -c "import robosuite as suite; env = suite.make('Lift', has_renderer=True); env.reset()"

# 2. 修改环境参数
# 改物体大小、颜色、起始位置等

# 3. 尝试自定义奖励函数
# 理解奖励塑形如何影响学习
```

---

## 🔑 总结

| 问题 | 答案 |
|------|------|
| **是什么** | 所有机器人操作任务的环境定义 |
| **有多少任务** | 9 个单臂 + 4 个双臂 = 13 个任务 |
| **核心类** | `ManipulationEnv`（基类） |
| **怎么用** | `suite.make("Lift", robots="Panda", ...)` |
| **先学哪个** | `lift.py` → `stack.py` → `manipulation_env.py` |