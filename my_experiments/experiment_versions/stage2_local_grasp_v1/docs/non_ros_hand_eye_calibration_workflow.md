# 非 ROS 手眼标定流程

这份文档面向你当前的目标：

- 机器人：Panda
- 相机：Intel RealSense D435i
- 安装方式：腕部相机，属于 eye-in-hand
- 使用目的：把真机相机外参对齐到仿真，用于后续 sim-to-real

这份文档尽量不用 ROS 术语，重点回答：

1. 手眼标定到底在做什么
2. 不用 ROS，应该怎么做
3. 最后会得到什么参数
4. 能不能直接照搬现成开源项目的参数

## 1. 手眼标定到底在做什么

手眼标定的目标只有一个：

- 求出相机相对机械臂末端的固定变换

也就是：

- 相机相对末端平移了多少
- 相机相对末端旋转了多少

如果用符号写，就是要求：

- `T_ee_camera`

通俗讲：

- 眼睛装在手上
- 手一直在动
- 但眼睛相对手的位置是不变的
- 我们就是把这个“不变的安装关系”求出来

## 2. 不用 ROS 能不能做

可以。

ROS 只是常见工程平台，不是原理的一部分。

不用 ROS 也能做，只要你能拿到这两类数据：

1. 机器人每一帧的末端位姿
2. 相机每一帧看到标定板的位姿

然后把这些数据喂给手眼标定算法就行。

## 3. 最后会得到什么

最后你真正想保存的是：

- 平移：`x, y, z`
- 旋转：旋转矩阵或四元数

通常会保存成一个 JSON：

```json
{
  "frame_definition": {
    "robot_base": "panda_link0",
    "end_effector": "panda_hand",
    "camera": "camera_color_optical_frame",
    "target": "apriltag_board"
  },
  "ee_to_camera": {
    "translation_m": [0.0, 0.0, 0.0],
    "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]
  },
  "quality": {
    "num_samples": 16,
    "rotation_rmse_deg": null,
    "translation_rmse_m": null
  }
}
```

这份结果以后用在：

1. 真机视觉和控制坐标系对齐
2. MuJoCo 里设置 `eye_in_hand` 的 `pos` 和 `quat`

## 4. 你需要准备什么

### 硬件

- Panda 机械臂
- 固定好的 D435i
- AprilTag 板，或多个 AprilTag 组成的板
- 一个稳定支架，把 AprilTag 固定在工作区

### 软件

- 能读取 Panda 末端位姿的接口
- 能读取 RealSense 图像的接口
- 能检测 AprilTag 的程序
- `opencv-python`
- `numpy`

建议你最后用 OpenCV 的 `cv2.calibrateHandEye(...)` 来求解。

参考：

- OpenCV `calibrateHandEye`: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html

## 5. 标定前的原则

标定期间一定要满足：

- 相机固定不动
- 相机支架固定不动
- AprilTag 板固定不动

只允许变化的是：

- Panda 的姿态

如果中途你碰了相机支架，标定结果就失效。

## 6. 非 ROS 实际流程

### 第 1 步：固定相机和标定板

1. 把 D435i 固定在 Panda 手腕上
2. 拧紧支架，不要松动
3. 把 AprilTag 板固定在桌面或工作区
4. 保证从多个角度都能看到板

建议：

- 板不要太小
- 不要反光太严重
- 不要离相机太远
- 也不要近到只看到板的一角

### 第 2 步：确认相机内参可用

手眼标定之前，相机内参最好已经是已知的。

对于 D435i，通常可以直接读取 RealSense 驱动提供的内参：

- `fx`
- `fy`
- `cx`
- `cy`

如果你后面用 AprilTag 位姿解算，内参要参与位姿估计。

### 第 3 步：定义坐标系

你要先明确这 4 个坐标系：

- `base`
- `ee`
- `camera`
- `target`

推荐定义：

- `base`：Panda 基座坐标系
- `ee`：Panda 手腕或夹爪参考坐标系
- `camera`：D435i color optical frame
- `target`：AprilTag 板坐标系

最重要的是：

- 整个流程都要用同一套定义

### 第 4 步：采样姿态

让 Panda 在 AprilTag 板前面采样多个不同姿态。

经验建议：

- 至少 12 个姿态
- 更稳一点：15 到 20 个

姿态要有变化：

- 左右变化
- 高低变化
- 距离变化
- 腕部有一定旋转变化

不要只采一圈差不多的姿态。

每到一个姿态，记录一组样本。

### 第 5 步：每组样本记录什么

每个样本都要保存两样东西。

#### A. 机器人位姿

记录：

- `base -> ee`

也就是：

- Panda 基座到末端的 4x4 位姿变换

这个可以通过：

- 机器人控制接口
- 正运动学
- 或你现有 Panda SDK 接口

拿到。

#### B. 视觉位姿

记录：

- `camera -> target`

也就是：

- 相机看到 AprilTag 板的位姿

通常流程是：

1. 采集当前相机图像
2. 检测 AprilTag
3. 根据 tag 尺寸和相机内参，估计板位姿

### 第 6 步：把样本存成文件

建议你把每个样本存成 4x4 齐次矩阵。

比如：

```json
{
  "samples": [
    {
      "sample_id": 0,
      "T_base_ee": [[...], [...], [...], [...]],
      "T_camera_target": [[...], [...], [...], [...]]
    }
  ]
}
```

这样最不容易乱。

## 7. 怎么求解

求解时你要把数据拆成 OpenCV 要的格式。

OpenCV 的 `calibrateHandEye(...)` 需要：

- `R_gripper2base`
- `t_gripper2base`
- `R_target2cam`
- `t_target2cam`

也就是说：

- 从 `T_base_ee` 里拆出旋转和平移
- 从 `T_camera_target` 里拆出旋转和平移

然后调用：

```python
R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
    R_gripper2base,
    t_gripper2base,
    R_target2cam,
    t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI,
)
```

OpenCV 文档说明它求的是：

- `cam2gripper`

你要特别注意坐标系方向，不要把它和 `ee_to_camera`、`camera_to_ee` 混掉。

参考：

- OpenCV `calibrateHandEye`: https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- FR3 手眼标定原理：`AX = XB`
  https://fr3setup.readthedocs.io/en/latest/camera_calibration/

## 8. 推荐求解方法

OpenCV 支持多个方法。

第一版建议：

- 先用 `Tsai`
- 再用 `Park`
- 必要时再试 `Daniilidis`

你可以把不同方法都算一遍，然后比较：

- 平移是否接近
- 旋转是否接近
- 验证误差谁更小

## 9. 怎么验证结果

不要只看求出来的一个数值，要做验证。

### 最低限度验证

1. 看结果是不是数量级正常
2. 看相机是不是还在手腕附近
3. 看朝向是不是和支架肉眼看起来差不多

如果你得到的结果像下面这样，就值得警惕：

- 相机离末端 30 厘米以上
- 朝向明显反过来
- 每次重算结果差别很大

### 更好的验证

采两三组没有参与求解的新姿态，做留出验证：

1. 用求出的 `T_ee_camera`
2. 结合 `T_base_ee`
3. 看 AprilTag 在 base 下预测的位置是否一致

通俗讲：

- 用新姿态测一遍
- 看这组外参还能不能解释新数据

## 10. 推荐的最小 Python 实现结构

你不一定现在就写，但建议按这个结构组织。

### A. 采集脚本

功能：

- 读取机器人末端位姿
- 读取相机图像
- 检测 AprilTag
- 保存样本到 JSON

### B. 求解脚本

功能：

- 读取 JSON 样本
- 调 OpenCV `calibrateHandEye`
- 输出 `camera_to_ee` 或 `ee_to_camera`

### C. 验证脚本

功能：

- 读取标定结果
- 用新样本做验证
- 输出误差

## 11. 一个推荐的实操顺序

第一次建议按这个顺序：

1. 先采 15 个样本
2. 用 OpenCV `Tsai` 求一版
3. 做 sanity check
4. 再补 5 个姿态
5. 重算
6. 做留出验证
7. 最后把结果写成 JSON

## 12. 能不能直接用 `mvp_grasp` 的参数

结论先说：

- 不建议直接照搬

`mvp_grasp` 很值得参考的地方是：

- 它就是 Panda + wrist-mounted D435
- 仓库里有 `cad` 目录
- README 明确写了有 3D 可打印相机支架

参考：

- [mvp_grasp README](https://github.com/dougsm/mvp_grasp)

但你不能直接照抄它的相机外参，原因有 4 个：

1. 你未必用的是同一个支架
2. 你未必装在同一个位置
3. 你末端参考坐标系定义可能不一样
4. 你的相机 optical frame 定义可能不一样

通俗讲：

- 同样是 D435
- 同样是 Panda
- 只要支架和坐标系定义不同
- 外参就不能直接通用

### 那它能拿来干什么

可以拿来做：

- 支架设计参考
- 安装思路参考
- 参数数量级参考
- wrist camera 系统结构参考

### 什么情况下可以“部分参考”

如果你做到下面 3 点，才可以把它当“初值参考”：

1. 用了非常接近的支架结构
2. 用了相同的末端参考坐标系
3. 明确了相机 frame 定义一致

即使这样，也还是建议你重新标定。

### 我的建议

- 可以参考它的 CAD 和安装思路
- 不要直接照用它的外参参数
- 最好把它当成“先验范围”，不是最终答案

## 13. 标定完成后，接下来做什么

你标定完成后，下一步建议是：

1. 把结果存成 JSON
2. 把结果抄到 MuJoCo 的腕部相机 `pos / quat`
3. 跑一段真机和仿真对比，看看视角是否接近
4. 再考虑加入相机位姿小范围随机化

## 14. 参考资料

### 手眼标定与工具

- OpenCV `calibrateHandEye`
  - https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
- FR3 Setup: Camera Calibration
  - https://fr3setup.readthedocs.io/en/latest/camera_calibration/
- MoveIt Hand-Eye Calibration Tutorial
  - https://moveit.github.io/moveit_tutorials/doc/hand_eye_calibration/hand_eye_calibration_tutorial.html
- `franka_easy_handeye`
  - https://github.com/franzesegiovanni/franka_easy_handeye

### Panda + wrist D435 开源项目

- `mvp_grasp`
  - https://github.com/dougsm/mvp_grasp
  - README 说明它面向 Panda + wrist-mounted D435，并在 `cad` 目录提供 3D 可打印相机支架

### 仿真相机与 sim-to-real

- Isaac Lab Camera Documentation
  - https://isaac-sim.github.io/IsaacLab/v2.1.0/source/overview/core-concepts/sensors/camera.html
- Triple Regression for Sim2Real Adaptation in Human-Centered Robot Grasping and Manipulation
  - https://openreview.net/forum?id=XTIWhKApWe

## 15. 最后一句建议

如果你现在问“最值得先做的一件事是什么”，我的建议是：

- 不要先猜外参
- 先做一版 15 个样本的手眼标定
- 得到一份你自己系统的 `ee_to_camera` 结果

这会比直接抄别人项目参数可靠得多。
