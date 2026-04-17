# 当前主线模型入口

这个目录只做一件事：

把**现在推荐使用**的模型和输入，固定成稳定入口。

这样以后不需要在 `training_runs/completed/...` 里面反复翻。

## 当前推荐文件

- [stage_b_model_current.zip](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_b_model_current.zip)
  - 当前推荐 `B` 模型
  - 实际指向：
    - `stage_b_grasp_cube_small_20260411_120759/checkpoints/stage_b_grasp_20000_steps.zip`

- [stage_c_model_current.zip](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_c_model_current.zip)
  - 当前推荐 `C` 模型
  - 实际指向：
    - `stage_c_lift_continue_cube_small_20260407_091334/final_model_steps_110644.zip`

- [vision_input_current.json](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/vision_input_current.json)
  - 当前推荐视觉输入
  - 实际指向：
    - `examples/vision_input_realsense_d435i_cube_small.json`

## 当前推荐整条链路命令

```bash
cd /home/mfj/robosuite
source /home/mfj/mj_robot/bin/activate
python my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/run_staged_grasp_pipeline.py \
  --vision-input /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/vision_input_current.json \
  --camera-profile realsense_d435i \
  --stage-b-model /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_b_model_current.zip \
  --stage-c-model /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_c_model_current.zip \
  --episodes 1 \
  --sleep 0.03 \
  --stage-pause 1.5
```

## 当前推荐遥操作接管命令

```bash
cd /home/mfj/robosuite
source /home/mfj/mj_robot/bin/activate
python my_experiments/experiment_versions/stage2_local_grasp_v1/tools/teleop_to_staged_grasp_pipeline.py \
  --vision-input /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/vision_input_current.json \
  --camera-profile realsense_d435i \
  --stage-b-model /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_b_model_current.zip \
  --stage-c-model /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_c_model_current.zip \
  --episodes 1 \
  --sleep 0.03 \
  --stage-pause 1.5 \
  --device keyboard
```

## 说明

这里的文件只是入口，不是模型本体。

优点：

- 以后换更好的 `B` 或 `C`，只需要更新这里的链接
- 你的命令不需要再改一长串真实 run 路径
