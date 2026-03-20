# 环境（Environments）

环境是外部代码将与 **robosuite** 交互的主要 API 对象。每个环境对应一个机器人操作任务，并提供一个标准接口供智能体与环境交互。虽然 **robosuite** 可以支持来自不同机器人领域的环境，但当前版本专注于操作环境。

接下来，我们将描述如何创建环境、如何与环境交互，以及每个环境如何在 MuJoCo 物理引擎中创建一个仿真任务。我们将使用 `TwoArmLift`（双臂举起）环境作为每一节的运行示例。

---

## 创建环境

通过调用 `robosuite.make` 并传入任务名称和一组配置环境属性的参数来创建环境。下面提供几个不同用例的示例。

```python
import robosuite
from robosuite.controllers import load_controller_config

# 加载操作空间控制 (OSC) 的默认控制器参数
controller_config = load_controller_config(default_controller="OSC_POSE")

# 创建一个用于屏幕可视化的环境
env = robosuite.make(
    "TwoArmLift",
    robots=["Sawyer", "Panda"],             # 加载一个 Sawyer 机器人和一个 Panda 机器人
    gripper_types="default",                # 为每个机械臂使用默认夹爪
    controller_configs=controller_config,   # 每个机械臂使用 OSC 控制
    env_configuration="single-arm-opposed", # (仅双臂环境) 机械臂相对放置
    has_renderer=True,                      # 屏幕渲染
    render_camera="frontview",              # 可视化 "frontview" 相机
    has_offscreen_renderer=False,           # 无离屏渲染
    control_freq=20,                        # 应用动作的控制频率为 20 Hz
    horizon=200,                            # 每个回合在 200 步后终止
    use_object_obs=False,                   # 不需要观测值
    use_camera_obs=False,                   # 不需要观测值
)

# 创建一个用于从低维观测进行策略学习的环境
env = robosuite.make(
    "TwoArmLift",
    robots=["Sawyer", "Panda"],             # 加载一个 Sawyer 机器人和一个 Panda 机器人
    gripper_types="default",                # 为每个机械臂使用默认夹爪
    controller_configs=controller_config,   # 每个机械臂使用 OSC 控制
    env_configuration="single-arm-opposed", # (仅双臂环境) 机械臂相对放置
    has_renderer=False,                     # 无屏幕渲染
    has_offscreen_renderer=False,           # 无离屏渲染
    control_freq=20,                        # 应用动作的控制频率为 20 Hz
    horizon=200,                            # 每个回合在 200 步后终止
    use_object_obs=True,                    # 向智能体提供物体观测
    use_camera_obs=False,                   # 不向智能体提供图像观测
    reward_shaping=True,                    # 使用稠密奖励信号进行学习
)

# 创建一个用于从像素进行策略学习的环境
env = robosuite.make(
    "TwoArmLift",
    robots=["Sawyer", "Panda"],             # 加载一个 Sawyer 机器人和一个 Panda 机器人
    gripper_types="default",                # 为每个机械臂使用默认夹爪
    controller_configs=controller_config,   # 每个机械臂使用 OSC 控制
    env_configuration="single-arm-opposed", # (仅双臂环境) 机械臂相对放置
    has_renderer=False,                     # 无屏幕渲染
    has_offscreen_renderer=True,            # 图像观测需要离屏渲染
    control_freq=20,                        # 应用动作的控制频率为 20 Hz
    horizon=200,                            # 每个回合在 200 步后终止
    use_object_obs=False,                   # 不向智能体提供物体观测
    use_camera_obs=True,                    # 向智能体提供图像观测
    camera_names="agentview",               # 使用 "agentview" 相机进行观测
    camera_heights=84,                      # 图像高度
    camera_widths=84,                       # 图像宽度
    reward_shaping=True,                    # 使用稠密奖励信号进行学习
)
```

### 模块化设计

下面我们对一些关键字参数提供额外说明，以突出创建 **robosuite** 环境的模块化结构，以及配置不同环境特性是多么容易。

- **`robots`**：此参数可用于轻松实例化带有不同机械臂的任务。例如，我们可以通过传入 `robots=["Jaco", "Jaco"]` 将任务改为使用两个 "Jaco" 机器人。一旦环境被初始化，这些机器人（由 [Robot](../simulation/robot.html#robot) 类捕获）可以通过环境内的 `robots` 数组属性访问，即：`env.robots[i]` 表示环境中的第 `i` 个机械臂。

- **`gripper_types`**：此参数可用于轻松更换每个机械臂的不同夹爪。例如，假设我们想更换上面示例中机械臂的默认夹爪。我们可以传入 `gripper_types=["PandaGripper", "RethinkGripper"]` 来实现。注意，也可以使用单一类型来自动将相同的夹爪类型广播到所有机械臂。

- **`controller_configs`**：此参数可用于轻松更换每个机械臂的动作空间。例如，如果我们想使用关节速度而不是 OSC 来控制机械臂，我们可以在上面的示例中使用 `load_controller_config(default_controller="JOINT_VELOCITY")`。与 `gripper_types` 类似，这个值可以是每个机械臂特定的，也可以是广播到所有机械臂的单一配置。

- **`env_configuration`**：此参数主要用于双臂任务，用于轻松配置机器人如何相对于彼此定向。例如，在 `TwoArmLift` 环境中，我们可以传入 `env_configuration="single-arm-parallel"`，使机械臂彼此相邻放置，而不是相对放置。

- **`placement_initializer`**：此参数是可选的，但可用于指定自定义的 `ObjectPositionSampler` 来覆盖 MuJoCo 物体的默认起始状态分布。采样器负责在每个回合开始时（例如，当调用 `env.reset()` 时）为场景中的所有物体采样一组有效的、非碰撞的放置位置。

---

## 与环境交互

### 策略循环（Policy Loop）

```python
# 此示例假设环境已创建，并执行一次智能体 rollout
import numpy as np

def get_policy_action(obs):
    # 这里可以使用训练好的策略，但我们选择随机动作
    low, high = env.action_spec
    return np.random.uniform(low, high)

# 重置环境以准备 rollout
obs = env.reset()

done = False
ret = 0.
while not done:
    action = get_policy_action(obs)         # 使用观测值决定动作
    obs, reward, done, _ = env.step(action) # 执行动作
    ret += reward
print("rollout 完成，总回报为 {}".format(ret))
```

### 观测值（Observations）

**robosuite** 的观测值是字典，包含每个模态的键值对。这使得智能体能够轻松处理不同形状的模态（例如，扁平的本体感知观测值和像素观测值）。注意，任何以 `*-state` 结尾的观测值条目表示属于 `*` 模态的所有单独观测值的连接。下面列出了常用的观测值键。

- **`robot0_proprio-state`**, **`robot1_proprio-state`**：每个机械臂的本体感知观测值。这包括机械臂关节位置（使用 `sin` 和 `cos` 编码）、机械臂关节速度、末端执行器位姿、夹爪手指位置、夹爪手指速度。此模态的形状是扁平的（例如 `(N,)`）。

- **`object-state`**：特定于任务的物体观测值。例如，`TwoArmLift` 环境提供锅的位姿、每个把手的位置，以及每个机器人夹爪相对于每个把手的相对位置。此模态的形状是扁平的（例如 `(N,)`）。

- **`{camera_name}_image`**：名为 `camera_name` 的相机的图像观测值。此模态的形状为 `(H, W, 3)`，其中 `H` 和 `W` 分别是图像的高度和宽度。默认情况下，返回的图像约定是 MuJoCo 的原生 `opengl`（"翻转"）。这也可以通过 `macros.py` 中的 `IMAGE_CONVENTION` 宏设置为 `opencv` 约定（未翻转）。

- **`{camera_name}_depth`**：名为 `camera_name` 的相机的深度图像观测值。此模态的形状为 `(H, W)`，其中 `H` 和 `W` 分别是图像的高度和宽度。默认情况下，返回的图像约定是 MuJoCo 的原生 `opengl`（"翻转"）。这也可以通过 `macros.py` 中的 `IMAGE_CONVENTION` 宏设置为 `opencv` 约定（未翻转）。

- **`image-state`**：（可选）堆叠的图像观测值。注意，这默认是禁用的，可以通过 `macros.py` 中的 `CONCATENATE_IMAGES` 宏来切换。

### 奖励和终止（Rewards and Termination）

每个环境在每个环境类的 `reward` 方法中实现一个奖励函数。奖励可以是二元成功或失败奖励（如果当前状态是任务完成状态则非零），或者是稠密的、经过塑形的奖励，其被设计为在解决任务的轨迹上（大部分）非负且非递减。所使用的奖励函数由 `reward_shaping` 参数决定。用于计算稀疏奖励的二元成功检查在每个环境类的 `_check_success` 方法中实现。

重要的是，**robosuite** 环境在达到成功标准时不会终止，而是始终继续固定的时间步数，由 `horizon` 参数决定。这是机器人操作领域强化学习的标准设计决策。

我们通过 `TwoArmLift` 的奖励函数和成功标准提供一个示例。注意，为简单起见，我们提供函数别名而不是实际实现细节，以便逻辑易于理解：

对于成功标准，我们只需检查锅是否成功提升到桌面上方某个高度阈值以上，并相应地返回 `True` 或 `False`。

```python
def _check_success(self):
    pot_height = get_pot_height()
    table_height = get_table_height()
    return pot_height > table_height + 0.10
```

奖励函数稍微复杂一些。首先，我们将奖励变量初始化为 0，并从环境中获取相关传感器数据，检查锅是否倾斜。

```python
def reward(self, action=None):
    reward = 0
    pot_tilt = get_pot_tilt()

    # 检查锅是否倾斜超过 30 度
    cos_30 = np.cos(np.pi / 6)
    direction_coef = 1 if pot_tilt >= cos_30 else 0
```

接下来，我们首先检查是否完成了任务（锅被提升到桌面上方且没有过度倾斜），如果是，则应用未归一化的奖励。

```python
    if self._check_success():
        reward = 3.0 * direction_coef
```

否则，如果我们使用奖励塑形，我们只提供部分奖励，并计算适当的奖励。

```python
    elif self.reward_shaping:
        
        # 提升奖励（平滑值在 [0, 1.5] 之间）
        pot_height = get_pot_height()
        r_lift = min(max(pot_height - 0.05, 0), 0.15)
        reward += 10. * direction_coef * r_lift
        
        # 接近奖励（平滑值在 [0, 1] 之间）
        left_hand_handle_distance = get_left_distance()
        right_hand_handle_distance = get_right_distance()
        reward += 0.5 * (1 - np.tanh(10.0 * left_hand_handle_distance))
        reward += 0.5 * (1 - np.tanh(10.0 * right_hand_handle_distance))
        
        # 抓握奖励（离散值在 [0, 0.5] 之间）
        left_hand_handle_contact = is_left_contact()
        right_hand_handle_contact = is_right_contact()
        if left_hand_handle_contact:
            reward += 0.25
        if right_hand_handle_contact:
            reward += 0.5
```

最后，我们需要归一化奖励，然后在最终返回计算的奖励之前，如果指定了 `reward_scale`，则将其值重新缩放到 `reward_scale`。

```python
    if self.reward_scale is not None:
        reward *= self.reward_scale / 3.0
        
    return reward
```

---

## 任务模型（Task Models）

每个环境拥有自己的 `MJCF` 模型，该模型通过将机器人、工作空间和物体适当加载到仿真器中来设置 MuJoCo 物理仿真。这个 MuJoCo 仿真模型在每个环境的 `_load_model` 函数中通过创建 `Task` 类的实例来程序化实例化。

每个 `Task` 类实例拥有一个 `Arena` 模型、一个 `RobotModel` 实例列表和一个 `ObjectModel` 实例列表。这些是 **robosuite** 类，它们引入了一种有用的抽象，使得在 MuJoCo 中设计场景变得容易。每个 `Arena` 基于一个定义工作空间（例如，桌子或箱子）和相机位置的 XML。每个 `RobotModel` 是表示任意机器人的 MuJoCo 模型（对于 `ManipulationModel`，这代表有臂机器人，例如 Sawyer、Panda 等）。每个 `ObjectModel` 对应一个加载到仿真中的物理物体（例如，立方体、带把手的锅等）。

---

## 任务描述

虽然 **robosuite** 可以支持来自不同机器人领域的环境，但当前版本专注于操作环境（`ManipulationEnv`）。下面我们对每个环境提供简要描述。对于这些标准化环境的基准测试结果，请查看 [Benchmarking](../algorithms/benchmarking) 页面。

### 单臂任务（Single-Arm Tasks）

#### 方块举起（Block Lifting）

![env_lift](../images/env_lift.png)

- **场景描述**：一个立方体放置在单个机械臂前方的桌面上。
- **目标**：机械臂必须将立方体提升到某个高度以上。
- **起始状态分布**：立方体的位置在每个回合开始时随机化。

#### 方块堆叠（Block Stacking）

![env_stack](../images/env_stack.png)

- **场景描述**：两个立方体放置在单个机械臂前方的桌面上。
- **目标**：机器人必须将一个立方体放置在另一个立方体顶部。
- **起始状态分布**：立方体的位置在每个回合开始时随机化。

#### 抓取 - 放置（Pick-and-Place）

![env_pick_place](../images/env_pick_place.png)

- **场景描述**：四个物体放置在单个机械臂前方的箱子中。箱子旁边有四个容器。
- **目标**：机器人必须将每个物体放入其对应的容器中。此任务也有更简单的单物体变体。
- **起始状态分布**：物体的位置在每个回合开始时随机化。

#### 螺母装配（Nut Assembly）

![env_nut_assembly](../images/env_nut_assembly.png)

- **场景描述**：两个彩色桩（一个方形和一个圆形）安装在桌面上，两个彩色螺母（一个方形和一个圆形）放置在单个机械臂前方的桌子上。
- **目标**：机器人必须将方形螺母拧到方形桩上，将圆形螺母拧到圆形桩上。此任务也有更简单的单螺母 - 桩变体。
- **起始状态分布**：螺母的位置在每个回合开始时随机化。

#### 开门（Door Opening）

![env_door](../images/env_door.png)

- **场景描述**：一扇带把手的门安装在单个机械臂前方的自由空间中。
- **目标**：机械臂必须学会转动把手并打开门。
- **起始状态分布**：门的位置在每个回合开始时随机化。

#### 擦桌子（Table Wiping）

![env_wipe](../images/env_wipe.png)

- **场景描述**：一张带有白板表面和一些标记的桌子放置在单个机械臂前方，机械臂的手上安装了一个白板擦。
- **目标**：机械臂必须学会擦拭白板表面并清除所有标记。
- **起始状态分布**：白板标记在每个回合开始时随机化。

### 双臂任务（Two-Arm Tasks）

#### 双臂举起（Two Arm Lifting）

![env_two_arm_lift](../images/env_two_arm_lift.png)

- **场景描述**：一个带两个把手的大锅放在桌面上。两个机器人机械臂放置在桌子的同一侧或桌子的两端。
- **目标**：两个机器人机械臂必须各抓一个把手并一起举起锅，达到某个高度以上，同时保持锅水平。
- **起始状态分布**：锅的位置在每个回合开始时随机化。

#### 双臂 peg-in-hole（Two Arm Peg-In-Hole）

![env_two_arm_peg_in_hole](../images/env_two_arm_peg_in_hole.png)

- **场景描述**：两个机器人机械臂彼此相邻或相对放置。一个机械臂拿着一块中心有方形孔的板，另一个机械臂拿着一个长桩。
- **目标**：两个机器人机械臂必须协调将桩插入孔中。
- **起始状态分布**：初始机械臂配置在每个回合开始时随机化。

#### 双臂传递（Two Arm Handover）

![env_two_arm_handover](../images/env_two_arm_handover.png)

- **场景描述**：一把锤子放在一张窄桌子上。两个机器人机械臂放置在桌子的同一侧或桌子的两端。
- **目标**：两个机器人机械臂必须协调，使离锤子更近的机械臂拿起锤子并传递给另一个机械臂。
- **起始状态分布**：锤子的位置和大小在每个回合开始时随机化。