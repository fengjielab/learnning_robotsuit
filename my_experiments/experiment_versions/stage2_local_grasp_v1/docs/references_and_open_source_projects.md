# 论文与开源项目参考清单

这份文档整理的是我们前面讨论过、并且后续这套 `stage2_local_grasp_v1` 很可能会继续参考的论文、教程和开源项目。

整理目标：

- 帮你快速回看“为什么我们要做分层抓取，而不是全局端到端 RL”
- 帮你回看“别人怎么处理 wrist camera、sim-to-real、手眼标定、shared control”
- 帮你后面做真实 Panda + D435i 部署时，有一份统一索引可查

---

## 1. 视觉 sim-to-real RL / 分层抓取相关论文

### 1.1 Human2Sim2Robot

- 标题：Crossing the Human-Robot Embodiment Gap with Sim-to-Real RL using One Human Demonstration
- 链接：
  - OpenReview: https://openreview.net/forum?id=CgGSFtjplI
  - 项目页: https://human2sim2robot.github.io/
- 我们参考它的原因：
  - 强调 `pre-manipulation pose` 的重要性
  - 说明“先把手送到一个好起点，再让 RL 学最后一段”是合理路线
  - 很适合支撑你“遥操作先送近，再让局部策略接管”的想法

### 1.2 Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids

- 标题：Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids
- 链接：
  - OpenReview: https://openreview.net/forum?id=8DHSyMFLbB
  - PMLR: https://proceedings.mlr.press/v305/lin25c.html
- 我们参考它的原因：
  - 说明视觉 sim-to-real RL 是可行的，但难点很多
  - 提到 reward formulation、real-to-sim tuning、policy distillation、mixed object representation
  - 适合提醒我们：视觉 RL 不是不能做，而是要非常讲究 recipe

### 1.3 DexPoint

- 标题：DexPoint: Generalizable Point Cloud Reinforcement Learning for Sim-to-Real Dexterous Manipulation
- 链接：
  - OpenReview: https://openreview.net/forum?id=tJE1Yyi8fUX
  - 项目页: https://yzqin.github.io/dexpoint
- 我们参考它的原因：
  - 强调结构化视觉输入，例如 point cloud，而不是只靠原始 RGB
  - 提到 contact-based rewards
  - 很适合提醒我们：视觉表示越几何化，往往越容易用于局部抓取控制

### 1.4 Goal-Auxiliary Actor-Critic for 6D Robotic Grasping with Point Clouds

- 标题：Goal-Auxiliary Actor-Critic for 6D Robotic Grasping with Point Clouds
- 链接：
  - OpenReview: https://openreview.net/forum?id=jOSWHddP1fZ
- 我们参考它的原因：
  - 使用 egocentric camera + segmented point cloud 做闭环 6D 抓取
  - 不是让机器人“看个大图就硬抓”，而是把输入组织成更抓取友好的形式
  - 对我们理解 “A 阶段为什么更适合局部视觉 / 局部几何” 很有帮助

---

## 2. 视觉伺服、shared control、自动切换相关

### 2.1 Robust Adaptive Robotic Visual Servo Grasping with Guaranteed Field of View Constraints

- 标题：Robust Adaptive Robotic Visual Servo Grasping with Guaranteed Field of View Constraints
- 链接：
  - MDPI: https://www.mdpi.com/2076-0825/13/11/457
- 我们参考它的原因：
  - 说明接近阶段常常是“先把误差压小，再进入闭合”
  - 很适合支撑我们“自动进入 A 的工作区，再切局部抓取”的思路
  - 对“视觉应该先服务对正，而不是一上来就闭爪乱碰”很有启发

### 2.2 Shared Control Teleoperation

- 标题：A robotic shared control teleoperation method based on learning from demonstrations
- 链接：
  - SAGE: https://journals.sagepub.com/doi/10.1177/1729881419857428
- 我们参考它的原因：
  - 说明“人先操作，系统再辅助或自动接管”是成熟方向
  - 对你“遥操作先送近，再让局部策略接手”的整体使用方式非常贴近

---

## 3. 手眼标定 / eye-in-hand 标定参考

### 3.1 MoveIt Hand-Eye Calibration Tutorial

- 标题：Hand-Eye Calibration
- 链接：
  - MoveIt 官方教程: https://moveit.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html
- 我们参考它的原因：
  - 讲得非常完整，适合理解采样流程
  - 明确说明每个样本由 `base->ee` 和 `camera->target` 组成
  - 明确提到通常 12 到 15 个样本后精度会比较稳定

### 3.2 MoveIt Calibration Repo

- 标题：MoveIt Calibration
- 链接：
  - GitHub: https://github.com/moveit/moveit_calibration
- 我们参考它的原因：
  - 是官方手眼标定工具代码库
  - 如果以后你临时愿意借用 ROS 工具链，这个是最标准的入口之一

### 3.3 FR3 Setup Camera Calibration

- 标题：Camera Calibration
- 链接：
  - 文档: https://fr3setup.readthedocs.io/en/latest/camera_calibration/
- 我们参考它的原因：
  - 用很清楚的方式解释了 eye-in-hand 的 `AX = XB`
  - 直接把“手眼标定的目标是求 `ee -> camera` 外参”说得很清楚

### 3.4 FR3 MoveIt Calibration Walkthrough

- 标题：Using MoveIt Calibration
- 链接：
  - 文档: https://fr3setup.readthedocs.io/en/latest/camera_calibration/moveit.html
- 我们参考它的原因：
  - 更偏实操
  - 对采样时 frame 该怎么选、样本大概要多少、有直接参考价值

### 3.5 franka_easy_handeye

- 标题：franka_easy_handeye
- 链接：
  - GitHub: https://github.com/franzesegiovanni/franka_easy_handeye
- 我们参考它的原因：
  - 这是 Panda / FR3 + RealSense + AprilTag 的现成 eye-in-hand 标定项目
  - 虽然你现在想走非 ROS 路线，但它对 frame 命名、设备组合、采样组织仍然很有参考意义

---

## 4. 相机参数、相机随机化、sim-to-real 视角对齐

### 4.1 Isaac Lab Camera Docs

- 标题：Camera
- 链接：
  - Isaac Lab 文档: https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/core-concepts/sensors/camera.html
- 我们参考它的原因：
  - 说明相机 pose、focal length、type 等参数都能控制
  - 适合参考“仿真里哪些相机参数值得随机化”

### 4.2 Isaac Lab Pinhole Camera Pattern

- 标题：isaaclab.sensors.patterns — Pinhole Camera Pattern
- 链接：
  - Isaac Lab API: https://isaac-sim.github.io/IsaacLab/v2.0.0/source/api/lab/isaaclab.sensors.patterns.html
- 我们参考它的原因：
  - 明确提到 intrinsic matrix 可传入和随机化
  - 适合支撑“不要只对齐 pose，也可以考虑对齐 / 随机化内参”

### 4.3 Triple Regression

- 标题：Triple Regression for Sim2Real Adaptation in Human-Centered Robot Grasping and Manipulation
- 链接：
  - OpenReview: https://openreview.net/forum?id=XTIWhKApWe
- 我们参考它的原因：
  - 直接讨论 real / sim camera perspectives 的投影误差
  - 适合提醒我们：就算相机位置差一点，也会明显影响迁移

### 4.4 Sim2Real Grasp Pose Estimation for Adaptive Robotic Applications

- 标题：Sim2Real Grasp Pose Estimation for Adaptive Robotic Applications
- 链接：
  - ScienceDirect: https://www.sciencedirect.com/science/article/pii/S2405896323004676
- 我们参考它的原因：
  - 强调 domain randomization 对 sim2real 的价值
  - 更偏视觉检测 / 姿态估计，但对我们后面做视角和图像随机化很有借鉴意义

---

## 5. 开源项目 / 工程实现参考

### 5.1 mvp_grasp

- 标题：dougsm/mvp_grasp
- 链接：
  - GitHub: https://github.com/dougsm/mvp_grasp
- 我们参考它的原因：
  - 明确是 Franka / Panda + wrist-mounted RealSense D435
  - 仓库里有 `cad` 目录，可直接参考 3D 打印支架
  - 是我们讨论“D435i 支架能不能借用”和“安装位置值不值得参考”时最关键的开源项目
- 额外说明：
  - 可以非常参考它的安装思路和 CAD
  - 但不能直接照搬它的手眼外参参数
  - 原因是：支架细节、末端 frame 定义、实际相机安装偏移都可能不同

---

## 6. 这些参考分别帮我们解决什么问题

### 6.1 为什么不做全局端到端 RL

主要参考：

- Human2Sim2Robot
- Sim-to-Real RL for Vision-Based Dexterous Manipulation on Humanoids
- DexPoint

它们共同说明：

- 视觉 RL 可行，但端到端全局控制很难
- 接近、对正、接触、抬起最好分层处理
- 更结构化的输入和更好的初始化非常重要

### 6.2 为什么我们强调“遥操作先送近，再让 RL 接管”

主要参考：

- Human2Sim2Robot
- Shared Control Teleoperation
- Visual Servo Grasping

它们共同说明：

- 好的 pre-manipulation pose 能显著降低 RL 难度
- 自动切换或 shared control 是成熟思路
- 局部对正阶段比全局搜索阶段更适合用 RL

### 6.3 为什么我们要认真做 wrist camera 和手眼标定

主要参考：

- MoveIt Hand-Eye Calibration
- FR3 Camera Calibration
- franka_easy_handeye
- Triple Regression

它们共同说明：

- 相机不是“有就行”，而是 pose 和 frame 非常关键
- 真机 wrist camera 的安装偏差会直接影响 sim-to-real
- `ee -> camera` 外参必须自己重新标

### 6.4 为什么我们还要做相机随机化

主要参考：

- Isaac Lab Camera Docs
- Isaac Lab Pinhole Camera Pattern
- Triple Regression
- Sim2Real Grasp Pose Estimation

它们共同说明：

- 只训练一个死视角很容易过拟合
- pose、intrinsics、图像风格都可以成为 reality gap 的来源

---

## 7. 对当前项目最直接的落地启发

按照这些参考，当前项目最合理的路线是：

1. 全局接近不用 RL
2. 视觉分类先决定 `object_profile`
3. `object_profile` 决定阻抗模板和阶段配置
4. wrist RGB-D 从 `Stage A` 开始最重要
5. `Stage A` 主要负责“无接触对正，形成干净抓取起点”
6. 真机部署前必须做手眼标定
7. 真机部署前后再做小范围相机位姿随机化和少量再训练

---

## 8. 备注

- 这份清单只整理了我们前面明确提到过、且和当前项目最相关的参考。
- 如果后面你要扩展到圆柱、软物体、触觉或者更强的力控，再单独补一版“接触与阻抗控制参考清单”会更合适。
