# stage2_local_grasp_v1

当前主线已经收束成一条更贴近真机部署的流程：

1. 人工遥操作把夹爪送到物块正上方
2. 相机只负责识别物体类型，映射到 `object_profile + impedance_template`
3. `A` 不再训练，固定为 scripted top-down descend
4. `B` 负责闭夹爪并形成稳定抓取
5. `C` 负责抬起并保持

一句话总结：

**人负责送近，视觉负责认类型，模板负责定抓法，系统负责 `A(scripted) -> B -> C`。**

## 当前推荐入口

### 1. 真机优先手动接管流程

看这里：

- [manual_handoff_real_robot_flow.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/manual_handoff_real_robot_flow.md)

### 1.5 当前模型固定入口

看这里：

- [training_runs/current/README.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/README.md)

### 2. 直接运行完整 `A -> B -> C`

```bash
cd /home/mfj/robosuite
source /home/mfj/mj_robot/bin/activate
python my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/run_staged_grasp_pipeline.py \
  --vision-input /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/examples/vision_input_realsense_d435i_cube_small.json \
  --camera-profile realsense_d435i \
  --stage-b-model /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/completed/stage_b_grasp_cube_small_20260411_120759/checkpoints/stage_b_grasp_20000_steps.zip \
  --stage-c-model /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/completed/stage_c_lift_continue_cube_small_20260407_091334/final_model_steps_110644.zip \
  --episodes 1 \
  --sleep 0.03 \
  --stage-pause 1.5
```

### 3. 手动遥操作后按键接管

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

接管按键：

- `p`：开始 `A(scripted) -> B -> C`

## 当前阶段分工

### A

- 不再训练
- 不再使用 PPO
- 只做 scripted 正上方下降到预抓取高度

### B

- 负责闭夹爪
- 形成双侧接触
- 稳定抓住物块

当前推荐模型：

- [stage_b_grasp_20000_steps.zip](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/completed/stage_b_grasp_cube_small_20260411_120759/checkpoints/stage_b_grasp_20000_steps.zip)

### C

- 负责抬起并保持
- 当前成功标准是抬高至少 `0.02 m`，并连续保持 `3` 步

当前推荐模型：

- [final_model_steps_110644.zip](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/completed/stage_c_lift_continue_cube_small_20260407_091334/final_model_steps_110644.zip)

## 现在还保留的视觉作用

视觉不再参与低层控制。

视觉当前只负责：

- `object_class`
- `shape_tag`
- `size_tag`
- `detection_confidence`

然后映射到：

- `object_profile`
- `impedance_template`
- 分阶段目标参数

## 重要文件

- [FILE_GUIDE.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/FILE_GUIDE.md)
- [ipc_first_deployment_guide.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/ipc_first_deployment_guide.md)
- [manual_handoff_real_robot_flow.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/manual_handoff_real_robot_flow.md)
- [non_ros_hand_eye_calibration_workflow.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/non_ros_hand_eye_calibration_workflow.md)
- [visp_franka_d435_reference_mount.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/visp_franka_d435_reference_mount.md)
- [references_and_open_source_projects.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/references_and_open_source_projects.md)
- [paper_positioning_and_references.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/paper_positioning_and_references.md)

## 当前目录里什么是旧路线

`legacy_stage2/` 只作为历史归档保留，不是当前主线。

如果你现在继续开发，优先只看：

- `core/`
- `pipelines/`
- `tools/`
- `docs/manual_handoff_real_robot_flow.md`
