# stage2_local_grasp_v1 文件说明

这份说明只保留当前主线相关文件。

## 你现在最常用的文件

### 运行入口

- [pipelines/run_staged_grasp_pipeline.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/run_staged_grasp_pipeline.py)
  - 直接运行 `A(scripted) -> B -> C`

- [tools/teleop_to_staged_grasp_pipeline.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/tools/teleop_to_staged_grasp_pipeline.py)
  - 手动遥操作后按 `p` 接管

### 训练入口

- [pipelines/train_stage_b_grasp.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/train_stage_b_grasp.py)
  - 训练 `B`

- [pipelines/continue_stage_b_grasp.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/continue_stage_b_grasp.py)
  - 继续训练 `B`

- [pipelines/train_stage_c_lift.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/train_stage_c_lift.py)
  - 训练 `C`

- [pipelines/continue_stage_c_lift.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/continue_stage_c_lift.py)
  - 继续训练 `C`

### 可视化 / 单阶段查看

- [tools/use_scripted_stage_a.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/tools/use_scripted_stage_a.py)
  - 单独看 scripted `A`

- [tools/use_stage_b_grasp_model.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/tools/use_stage_b_grasp_model.py)
  - 单独看 `B`

- [tools/use_stage_c_lift_model.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/tools/use_stage_c_lift_model.py)
  - 单独看 `C`

## 底层核心文件

- [core/object_profiles.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/object_profiles.py)
  - 物体模板、阶段目标、阻抗模板都在这里

- [core/vision_interface.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/vision_interface.py)
  - 视觉 JSON 输入转成 `object_profile`

- [core/stage2_local_grasp_env.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/stage2_local_grasp_env.py)
  - 共用底座环境

- [core/stage_a_cage_env.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/stage_a_cage_env.py)
  - 当前主要提供 scripted `A` 的逻辑

- [core/stage_b_grasp_env.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/stage_b_grasp_env.py)
  - `B` 阶段抓取逻辑

- [core/stage_c_lift_env.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/core/stage_c_lift_env.py)
  - `C` 阶段抬升逻辑

## 视觉输入示例

- [examples/vision_input_realsense_d435i_cube_small.json](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/examples/vision_input_realsense_d435i_cube_small.json)
- [examples/vision_input_realsense_d435i_cube_small_loose.json](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/examples/vision_input_realsense_d435i_cube_small_loose.json)

当前推荐主线先用：

- `cube_small`

## 当前推荐文档

- [README.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/README.md)
- [docs/ipc_first_deployment_guide.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/ipc_first_deployment_guide.md)
- [docs/manual_handoff_real_robot_flow.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/manual_handoff_real_robot_flow.md)
- [docs/non_ros_hand_eye_calibration_workflow.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/non_ros_hand_eye_calibration_workflow.md)
- [docs/visp_franka_d435_reference_mount.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/visp_franka_d435_reference_mount.md)
- [docs/references_and_open_source_projects.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/references_and_open_source_projects.md)
- [docs/paper_positioning_and_references.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/paper_positioning_and_references.md)

## 当前不再建议使用的旧主线

下面这些思路已经退出当前主线：

- 视觉直接参与 `A/B/C` 的低层控制
- 单独继续训练 learned `A`
- 旧版 `teleop -> auto handoff to learned A`

所以现在目录里不再保留对应脚本和说明，避免继续混淆。
