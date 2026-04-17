# 论文定位与参考文献

这份文档只围绕当前主线整理：

1. 人工遥操作把夹爪送到物块正上方
2. 相机只负责识别物体类型
3. 识别结果映射到 `object_profile + impedance_template`
4. `A` 使用 scripted top-down descend
5. `B` 使用学习到的抓取闭合策略
6. `C` 使用学习到的抬升保持策略

这不是“全局端到端 RL”，而是一条更偏 **半自主抓取系统 / 分层抓取系统 / human-in-the-loop grasping system** 的路线。

---

## 1. 这条路线能不能发 SCI

结论先说：

**能，但更像系统型 / 工程型 / 应用型 SCI 论文，不像纯算法突破型论文。**

如果只是把很多模块堆在一起跑通，通常还不够。
要想发出去，至少要把它讲成下面三类中的一种：

### 1.1 系统型论文

重点讲：

- 人工送近 + 自动接管的分层抓取流程
- 分类驱动的模板切换
- scripted `A` + learned `B/C` 的阶段式抓取系统
- 相比“全局端到端 RL”更安全、更稳、更适合真机

适合的前提：

- 有完整系统架构
- 有真实机器人实验
- 有失败模式分析

### 1.2 应用型论文

重点讲：

- 面向某类物体或某类任务的抓取系统
- 不同物体类型对应不同阻抗模板
- 在真实平台上提高稳定性、降低损伤、减少碰桌或减少人工负担

适合的前提：

- 有明确应用场景
- 有物体类别划分和模板切换实验
- 有任务成功率和安全性指标

### 1.3 方法型论文

如果想更像方法论文，不能只说“我们把这些模块接起来了”，还要突出一个清晰的方法点，比如：

- 一种 **human-guided top-down handoff** 机制
- 一种 **category-conditioned impedance template selection** 机制
- 一种 **scripted approach + learned contact-rich closure/lift** 的混合策略框架

---

## 2. 当前方案更适合怎么讲

不建议讲成：

- 一个大而全的端到端 RL 系统
- 一个纯视觉强化学习抓取方法

更建议讲成：

**一种面向真机部署的、类别条件驱动的半自主分层抓取框架。**

可以压成一句英文定位：

`A category-conditioned, human-in-the-loop staged grasping framework for real-world deployment.`

这个定位有几个优点：

- 和你现在真实系统一致
- 不需要硬吹“纯算法创新”
- 容易解释为什么不用全局端到端 RL
- 容易解释为什么视觉只做分类
- 容易解释为什么阻抗模板要按类别切换

---

## 3. 如果要发 SCI，最需要补什么

### 3.1 必须有真实机器人实验

这是最重要的。

如果没有真机，只在 MuJoCo 里跑通：

- 更像技术报告
- 论文说服力会明显不够

最少建议：

- 至少一类物体真实抓取
- 至少几种大小或几种类别
- 给出多次重复实验成功率

### 3.2 必须有基线比较

至少要和下面几类基线比较一部分：

- 全手动遥操作
- 不做模板切换的统一阻抗
- 不做 staged pipeline 的直接抓取
- 不做人工送近的更自动但更脆弱方案

### 3.3 必须有消融

最少建议做这些消融：

- 去掉分类切模板
- 去掉 scripted `A`
- 去掉 category-conditioned impedance
- 不同 handoff 方式对比

### 3.4 必须有失败分析

比如：

- `B` 为什么会双侧接触失败
- 为什么会碰桌
- 哪类物体最容易失效
- 人工送近偏差会如何影响系统

---

## 4. 和我们最接近的参考方向

下面这些不是和我们“完全一样”，但和我们现在的主线高度相关。

### 4.1 Shared autonomy / 人工送近再自动接手

#### A robotic shared control teleoperation method based on learning from demonstrations

- 链接：https://journals.sagepub.com/doi/10.1177/1729881419857428
- 类型：shared control / teleoperation assistance
- 和我们的关系：
  - 支持“人工先送近，再由系统辅助或接管”这条大方向
  - 说明 shared autonomy 本身就是成熟问题，不是拍脑袋想出来的

### 4.2 预操作位姿很重要

#### Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration

- 链接：https://openreview.net/forum?id=CgGSFtjplI
- 类型：CoRL 2025
- 和我们的关系：
  - 强调 `pre-manipulation pose` 很关键
  - 支持“先把手送到好起点，再做后续接触动作”的思想

### 4.3 分阶段 / pick-place 风格流程

#### MoveIt Pick and Place Tutorial

- 链接：https://docs.ros.org/en/melodic/api/moveit_tutorials/html/doc/pick_place/pick_place_tutorial.html
- 类型：工程教程 / 经典抓取流程
- 和我们的关系：
  - `pre_grasp_approach -> grasp posture -> post_grasp_retreat`
  - 非常像我们现在的 `A -> B -> C`

#### MoveIt Task Constructor Pick and Place

- 链接：https://moveit.github.io/moveit_task_constructor/tutorials/pick-and-place.html
- 类型：任务分阶段构造
- 和我们的关系：
  - 更明确地把“靠近 / 抓取 / 抬升”拆开
  - 支持 staged pipeline 的表述方式

### 4.4 类别驱动抓取

#### Category-based task specific grasping

- 链接：https://www.sciencedirect.com/science/article/pii/S0921889015000846
- DOI：https://doi.org/10.1016/j.robot.2015.04.002
- 类型：类别驱动抓取
- 和我们的关系：
  - 支持“先判断物体属于哪一类，再决定抓法”
  - 虽然它不是阻抗模板论文，但和我们“分类 -> 模板”思路很近

#### An adaptive planning framework for dexterous robotic grasping with grasp type detection

- 说明：这是我们讨论时的相近方向，但当前我没有重新核对到一个更干净的开放链接，写正文时建议你再补正式出处。
- 和我们的关系：
  - 支持“抓取类型 / 物体类型影响具体抓法”

### 4.5 结构化视觉而不是直接端到端 RGB

#### DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation

- 链接：https://openreview.net/forum?id=tJE1Yyi8fUX
- 参考页：https://researchportal.hkust.edu.hk/en/publications/dexpoint-generalizable-point-cloud-reinforcement-learning-for-sim-2
- 类型：sim-to-real dexterous manipulation
- 和我们的关系：
  - 说明结构化表示对抓取更友好
  - 也支持我们后来不再让原始 RGB 直接控制低层策略

### 4.6 视觉 RL 可以做，但 recipe 很重

#### Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids

- 链接：https://proceedings.mlr.press/v305/lin25c.html
- 类型：CoRL 2025
- 和我们的关系：
  - 说明视觉 RL 是可以做的
  - 但需要复杂 recipe、调参与分阶段处理
  - 也反过来支持我们为什么收缩成“视觉只做分类”

### 4.7 不同物体需要不同阻抗 / 柔顺参数

#### Research on the application of impedance control in flexible grasp of picking robot

- 链接：https://journals.sagepub.com/doi/10.1177/16878132231161016
- 类型：柔顺抓取 / 阻抗控制
- 和我们的关系：
  - 支持“阻抗参数不是固定死的”
  - 支持不同物体条件下需要不同抓取柔顺性

#### Grasping Force Optimization and DDPG Impedance Control for Apple Picking Robot End-Effector

- 链接：https://www.mdpi.com/2077-0472/15/10/1018
- 类型：抓取力与阻抗控制
- 和我们的关系：
  - 支持“物体属性影响抓取力和阻抗设置”
  - 可以帮助你在论文里解释为什么需要 `impedance_template`

### 4.8 Panda + wrist D435 的工程参考

#### dougsm/mvp_grasp

- 链接：https://github.com/dougsm/mvp_grasp
- 类型：开源工程系统
- 和我们的关系：
  - Panda + wrist-mounted D435
  - 适合作为工程安装与系统实现参考
  - 不是和我们同一路线，但非常适合做工程对照

### 4.9 官方 Franka + D435 标定参考

#### ViSP / visp_ros Franka tutorial

- 链接：https://docs.ros.org/en/noetic/api/visp_ros/html/tutorial-franka-coppeliasim.html
- 和我们的关系：
  - 给出了 Franka + D435 支架和 `eMc` 标定参考
  - 适合在论文中说明你的安装与标定参考来源

---

## 5. 论文里可以怎么写创新点

如果按当前路线，我建议创新点不要写得太虚。

比较稳的写法是：

### 贡献 1

提出了一种面向真实部署的 **human-in-the-loop staged grasping framework**：

- 人工送近
- 视觉分类
- scripted approach
- learned closure and lift

### 贡献 2

提出了一种 **category-conditioned grasp template selection** 机制：

- 不同物体类型映射到不同 `object_profile`
- 不同 `object_profile` 使用不同阻抗模板和阶段参数

### 贡献 3

验证了相比更大范围的统一控制策略，这种分层方式在：

- 成功率
- 安全性
- 碰桌率
- 真机可部署性

上更有优势。

---

## 6. 现在这条路线最适合投什么风格

更像：

- 系统型机器人论文
- 工程应用型机器人论文
- 面向真实抓取部署的集成论文

不太像：

- 顶会级纯 RL 算法论文
- 纯视觉策略学习论文
- “一个新网络结构解决一切”的论文

所以，如果你问：

**“我们这样的一锅炖能不能发 SCI？”**

我的判断是：

**能，但前提是你把它写成一个清楚的系统方法，并用真实实验说明这条分层路线为什么比更重、更脆的方案更适合部署。**

如果只有“我们把几个模块接起来能跑”，那还不够。

---

## 7. 论文实验建议

建议最少做下面这些：

### 7.1 主实验

- 完整流程成功率
- `A -> B -> C` 各阶段成功率
- 真实机器人实验

### 7.2 消融实验

- 无分类模板切换
- 无阻抗模板切换
- 无 scripted `A`
- 无人工送近，只做更自动化入口

### 7.3 安全性实验

- 碰桌率
- 抓取失败率
- 释放或掉落率

### 7.4 泛化实验

- 不同尺寸
- 不同类别
- 不同光照 / 背景

---

## 8. 你现在最该怎么定位自己这篇文章

最推荐的定位不是：

- “我们提出了一个全新强化学习算法”

而是：

**“我们提出了一个面向真实抓取部署的、类别条件驱动的人机协同分层抓取框架。”**

这和你当前真正做出来的系统是最一致的。

