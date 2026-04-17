# ViSP Franka + D435 官方参考安装外参

这份说明记录的是 **ViSP / visp_ros 官方教程代码里给出的默认 `eMc` 外参**。

它对应的是：

- 机器人：Franka Panda
- 相机：Intel RealSense D435
- 支架：`franka-rs-D435-camera-holder.stl`

来源：

- 教程页：
  - https://docs.ros.org/en/melodic/api/visp_ros/html/tutorial-franka-coppeliasim.html
- 示例代码：
  - https://docs.ros.org/en/noetic/api/visp_ros/html/tutorial-franka-real-pbvs-apriltag_8cpp-example.html

## 官方代码中的默认 `ePc`

在 `tutorial-franka-real-pbvs-apriltag.cpp` 里，ViSP 官方给出的默认值是：

```text
ePc[0] = 0.0564668
ePc[1] = -0.0375079
ePc[2] = -0.150416
ePc[3] = 0.0102548
ePc[4] = -0.0012236
ePc[5] = 1.5412
```

这里：

- 前 3 个值是平移 `tx, ty, tz`，单位米
- 后 3 个值是 ViSP 的 `theta-u` 旋转表示

## 已换算成我们配置里更方便使用的形式

我们在 [camera_profiles.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/camera_profiles.py) 里记录为：

```python
translation_m = [0.0564668, -0.0375079, -0.150416]
quaternion_xyzw = [
    0.004634771931637911,
    -0.0005530197503171342,
    0.6965626341850011,
    0.7174808079074659,
]
```

## 重要提醒

这组值只能当：

- 官方参考安装位姿
- sim 对齐的初始参考
- 你自己支架设计和相机朝向的 sanity check

不能直接当：

- 你自己真机最终 `ee_to_camera`

原因很简单：

- 支架打印误差会变
- 安装姿态会变
- 末端参考 frame 可能定义不同
- D435 / D435i 的实际安装细节也会有差别

所以最终上真机前，仍然建议你做自己的手眼标定，然后再覆盖这组参考值。

## 在 MuJoCo 里如何真正落地

我们当前项目里保留了两层数据：

- 官方原始参考 `eMc`
- 已匹配到 robosuite `robot0_right_hand` 相机节点的本地位姿

原因是：

- 官方 `eMc` 的末端参考 frame 与 robosuite Panda 的 `eye_in_hand` 相机父 body 不是同一个 frame
- 直接把官方四元数和平移写进 MuJoCo，会出现“相机看天 / 看错方向”的问题

因此，我们在 [camera_profiles.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/camera_profiles.py) 里额外保存了一组已经验证过的 MuJoCo 本地位姿：

```python
validated_mujoco_local_mount = {
    "cam_pos_m": [0.0564668, -0.053416, 0.0375079],
    "cam_quat_wxyz": [-0.00288623, 0.99987971, -0.01479138, 0.00366832],
}
```

这组值的含义是：

- 仍然来源于官方 Franka + D435 参考安装
- 但已经补上了我们项目里 Panda wrist camera 的 frame 匹配
- 这是当前会被 [stage2_local_grasp_env.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/stage2_local_grasp_env.py) 真正写入 MuJoCo `cam_pos / cam_quat` 的值
