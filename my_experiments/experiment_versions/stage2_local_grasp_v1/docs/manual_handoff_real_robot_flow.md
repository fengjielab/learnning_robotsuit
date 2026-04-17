# 真机优先手动接管流程

## 目标

先用最稳的方式把整条链路跑通：

1. 人工遥操作把夹爪送到物块正上方
2. 人工按键确认接管
3. 系统执行 `A(scripted) -> B -> C`

这版默认不依赖自动接管判断，优先保证：
- 安全
- 可控
- 好调试

## 当前推荐流程

### Step 1: 启动手动接管版流水线

```bash
cd /home/mfj/robosuite
source /home/mfj/mj_robot/bin/activate
python my_experiments/experiment_versions/stage2_local_grasp_v1/tools/teleop_to_staged_grasp_pipeline.py \
  --vision-input /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/examples/vision_input_realsense_d435i_cube_small.json \
  --camera-profile realsense_d435i \
  --stage-b-model /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/completed/stage_b_grasp_cube_small_20260411_120759/checkpoints/stage_b_grasp_20000_steps.zip \
  --stage-c-model /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/completed/stage_c_lift_continue_cube_small_20260407_091334/final_model_steps_110644.zip \
  --episodes 1 \
  --sleep 0.03 \
  --stage-pause 1.5 \
  --device keyboard
```

### Step 2: 人工遥操作靠近

目标不是“先碰到物块”，而是：

- 夹爪张开
- 夹爪中心基本在物块正上方
- 左右前后误差尽量小
- 高度在物块上方几厘米
- 不碰桌
- 不碰物块

通俗讲：

先把手送到“可以安全往下抓”的位置，再交给系统。

### Step 3: 人工按 `p` 接管

当前推荐只用手动接管。

- 当你觉得位置已经合适时，按 `p`
- 系统会进入：
  - `A(scripted)`：沿 z 轴下降到预抓取高度
  - `B`：闭夹爪抓取
  - `C`：抬起并保持

## 各阶段作用

### A 阶段

不再训练，不再让 PPO 学。

现在的 `A` 是规则化阶段，只做：

- 从正上方向下
- 降到预抓取高度
- 为 `B` 提供稳定抓取入口

### B 阶段

负责：

- 闭夹爪
- 形成双侧接触
- 稳定抓住物块

当前推荐模型：

- [stage_b_grasp_20000_steps.zip](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/completed/stage_b_grasp_cube_small_20260411_120759/checkpoints/stage_b_grasp_20000_steps.zip)

### C 阶段

负责：

- 抬起
- 保持抓住
- 满足成功高度

当前推荐模型：

- [final_model_steps_110644.zip](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/completed/stage_c_lift_continue_cube_small_20260407_091334/final_model_steps_110644.zip)

## 为什么先用手动接管

因为这最接近真机第一版的安全策略：

- 人决定什么时候开始抓
- 系统决定怎么完成抓

好处：

- 不会因为自动接管时机判错而提前动作
- 更容易判断失败到底是“靠近没到位”，还是 `B/C` 本身的问题
- 更适合真机调试

## 当前不做的事

当前默认不把“自动接管”作为真机主流程。

原因：

- 旧的自动接管门槛是通用近距离门槛
- 还不是专门为“正上方下降”流程设计的

后续如果要做自动接管，应该改成：

- 正上方对齐
- 安全高度窗口
- 张开
- 无接触

这种“正上方接管条件”。

## 一句话总结

当前真机优先版流程就是：

人工把夹爪送到物块正上方 -> 人工按 `p` 接管 -> 系统执行 `A(scripted) -> B -> C`。
