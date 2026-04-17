# MuJoCo 腕部相机参数说明

这份文档只讲一件事：

在当前 `stage2_local_grasp_v1` 里，MuJoCo 的腕部相机到底是靠哪些参数起作用的，以及它们分别会影响什么。

适用场景：

- 你想知道“为什么真机相机装高一点、低一点，策略可能就不好用了”
- 你想知道“现在代码里哪些参数真正在控制相机”
- 你未来想把仿真相机调得更像真实 Panda + D435i

## 一句话总览

当前腕部相机效果，主要由下面几类参数决定：

1. 相机装在哪里：`pos`
2. 相机朝哪看：`quat`
3. 相机看多宽：`fovy`
4. 图像输出多大：`camera_height` / `camera_width`
5. 要不要深度图：`camera_depths`
6. 深度怎么变成米：`znear` / `zfar` 和 `get_real_depth_map(...)`

如果未来真机上的 D435i 安装位姿和仿真差很多，最先出问题的通常就是第 1 和第 2 项。

## 1. 相机在 MuJoCo 里的定义位置

Panda 末端腕部相机定义在：

- [robot.xml](/home/mfj/robosuite/robosuite/models/assets/robots/panda/robot.xml#L237)

当前 XML 是：

```xml
<camera mode="fixed" name="eye_in_hand" pos="0.05 0 0" quat="0 0.707108 0.707108 0" fovy="75"/>
```

这里最重要的 3 个参数是：

- `pos`
- `quat`
- `fovy`

## 2. `pos` 是什么

`pos="0.05 0 0"`

它表示：

- 相机相对它所在那个 body 的局部位置偏移

在这个 Panda XML 里，这只相机挂在 `right_hand` body 上，所以这里的 `pos` 说的是：

- 相机相对手腕末端 body，往前后左右上下偏了多少

通俗讲：

- `x` 变大：相机更往前伸
- `x` 变小：相机更往后缩
- `y` 变化：相机左右偏
- `z` 变化：相机上下偏

这就是最直接影响“真机装高了还是装低了”的参数。

## 3. `quat` 是什么

`quat="0 0.707108 0.707108 0"`

它表示：

- 相机在局部坐标系下的朝向

通俗讲：

- 镜头到底往哪看
- 有没有往下压一点
- 有没有往左歪一点
- 有没有滚转一点

如果 `quat` 和真实 D435i 的安装朝向不一样，那么即使 `pos` 差不多，看到的画面也会不一样。

对 sim-to-real 来说，`quat` 和 `pos` 一样重要。

## 4. `fovy` 是什么

`fovy="75"`

它表示：

- 相机的竖直视场角

通俗讲：

- 数值更大：更广角，能看到更多范围，但物体看起来更小
- 数值更小：更窄，能看到的范围更小，但目标更“放大”

如果真机 D435i 的实际视场和这里差很多，策略看到的视觉分布也会变。

## 5. 代码里实际用的是哪只相机

当前实验代码固定使用：

- [stage2_local_grasp_env.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/stage2_local_grasp_env.py#L96)

也就是：

```python
self.camera_name = "robot0_eye_in_hand"
```

通俗讲：

- 训练里用的是 Panda 末端那只腕部相机
- 不是前视角，不是鸟瞰，不是桌边固定相机

## 6. 图像大小和深度开关在哪里控制

在环境创建时：

- [stage2_local_grasp_env.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/stage2_local_grasp_env.py#L183)

这里真正起作用的是：

```python
camera_names=[self.camera_name]
camera_heights=[self.camera_height]
camera_widths=[self.camera_width]
camera_depths=[True]
```

它们分别表示：

- `camera_names`
  - 用哪只相机
- `camera_heights`
  - 图像高度
- `camera_widths`
  - 图像宽度
- `camera_depths`
  - 是否同时输出 depth

通俗讲：

- 这里决定“拍多大”
- 也决定“是不是 RGB-D，而不只是 RGB”

## 7. D435i profile 目前真正影响了什么

当前 D435i profile 在：

- [camera_profiles.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/camera_profiles.py#L6)

现在它主要提供：

- `640 x 480`
- 深度最小距离 `0.105 m`
- 深度最大距离 `10.0 m`
- 颜色流和深度流的默认元数据

这些值会被读进：

- [stage2_local_grasp_env.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/stage2_local_grasp_env.py#L91)

所以当前 `realsense_d435i` profile 真正控制的主要是：

- 输出分辨率
- 深度范围参考
- 视觉特征编码时的裁剪和归一化基准

要注意：

- 现在还没有把 D435i 的真实内参 `fx / fy / cx / cy` 真正灌进 MuJoCo 相机
- 也就是说，现在更像“D435i 风格配置”，不是“精确 D435i 光学模型”

## 8. 深度图为什么还要再转一次

MuJoCo 原始返回的 depth 不是直接米制深度。

当前代码会先在：

- [camera_obs.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/vision/camera_obs.py#L27)

调用：

```python
get_real_depth_map(sim=sim, depth_map=depth)
```

真正的变换逻辑在：

- [camera_utils.py](/home/mfj/robosuite/robosuite/utils/camera_utils.py#L120)

这里还会用到：

- `sim.model.vis.map.znear`
- `sim.model.vis.map.zfar`

通俗讲：

- 深度图不是简单一张灰度图
- 它还依赖 MuJoCo 的近裁剪面和远裁剪面
- 这些参数会影响“0 到 1 的深度值”怎么被翻译成“几米”

## 9. 哪些参数最影响未来真机迁移

如果目标是未来迁移到真实 Panda + D435i，优先级大致是：

1. `pos`
2. `quat`
3. `fovy`
4. 分辨率
5. 深度范围和深度噪声

通俗讲：

- 最先要对齐的是“相机装在哪”和“相机往哪看”
- 然后才是“看多宽”
- 再后面才是“图像多大”和“深度噪声像不像”

## 10. 哪些参数改了以后，画面会怎么变

### 改 `pos`

影响：

- 物体在图像中的位置
- 物体和夹爪的相对视觉关系
- 看起来离目标近了还是远了

风险：

- 策略会感觉像“换了一只眼睛”

### 改 `quat`

影响：

- 物体在图像中是偏上偏下还是偏左偏右
- 夹爪和物体的相对角度模式

风险：

- 很容易让策略认不出它之前学过的视觉模式

### 改 `fovy`

影响：

- 目标在图像里看起来更大还是更小
- 看到的背景范围有多大

风险：

- 尺度感会变

### 改分辨率

影响：

- 图像更细还是更糙
- 编码器吃到的纹理和边缘细节

风险：

- 训练和部署分辨率不同，会导致输入分布变化

### 改 `znear / zfar`

影响：

- depth 转米后的数值分布
- 近处和远处的深度精度感

风险：

- 深度特征会变

## 11. 你现在最值得优先关注什么

如果你马上要做 sim-to-real，对当前代码来说最值得优先关注的是：

1. Panda XML 里的相机 `pos`
2. Panda XML 里的相机 `quat`
3. Panda XML 里的相机 `fovy`
4. 当前 `640 x 480` 和真实 D435i 是否一致

也就是说：

- 先把“装在哪”和“朝哪看”对齐
- 再谈更细的视觉拟合

## 12. 当前这套系统的通俗总结

你现在的系统可以理解成：

- MuJoCo 里有一只装在 Panda 手腕上的虚拟相机
- 这只相机的位置和朝向由 XML 控制
- 训练时图像大小和是否带深度由环境参数控制
- 深度值再由 MuJoCo 的 near / far 参数转换成米制

所以如果未来真机上 D435i 的安装位姿变了，最应该先查的不是 PPO，也不是 reward，而是：

- 仿真相机 `pos`
- 仿真相机 `quat`
- 仿真相机 `fovy`

## 13. 后续建议

如果下一步要继续往真机迁移走，建议按这个顺序处理：

1. 先测量真实 D435i 相对 Panda 手腕的安装位姿
2. 把 XML 里的 `pos / quat` 调到尽量接近
3. 检查 `fovy` 是否接近真实相机视角
4. 训练时再加入少量相机位姿随机化

这样更容易让仿真里学到的东西在真机上还能用。
