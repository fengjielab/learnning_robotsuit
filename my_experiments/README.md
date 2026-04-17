# my_experiments

这个目录现在按用途整理成了 6 个主要入口，避免教程、训练脚本、实验版本和模型产物混在一起。

## 当前结构

```text
my_experiments/
├── README.md
├── docs/                              说明文档和笔记
├── tutorials/                         入门探索脚本和自测脚本
├── baseline_lift_ppo/                 最早一版 Lift + PPO 基线实验
├── experiment_versions/
│   ├── 2026-03-23_stable_training/    0323 稳定版训练
│   ├── 2026-03-23_render_training/    0323 实时渲染训练
│   └── stage2_local_grasp_v1/         二阶段局部抓取训练（当前主线）
└── archive/
    └── duplicate_root_copies/         从根目录清走的重复脚本备份
```

## 你现在最常用的入口

### 1. 当前主线：二阶段局部抓取训练

目录：

`/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1`

先验证 reset：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1
source /home/mfj/mj_robot/bin/activate
python validate_stage2_reset.py --num-resets 100
```

开始训练：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1
source /home/mfj/mj_robot/bin/activate
python train_stage2_local_grasp.py --total-timesteps 300000
```

继续训练：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1
source /home/mfj/mj_robot/bin/activate
python continue_stage2_local_grasp.py --additional-timesteps 200000
```

这条线默认预留了真实视觉标签接口，相机型号默认按 `RealSense D435i` 写。

### 2. 实时渲染训练

目录：

`/home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_render_training`

运行：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_render_training
source /home/mfj/mj_robot/bin/activate
python train_with_render_0323_2045.py
```

### 3. 稳定版继续训练

目录：

`/home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training`

运行：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
source /home/mfj/mj_robot/bin/activate
python continue_training_v3_0323.py
```

### 4. 最早的基线实验

目录：

`/home/mfj/robosuite/my_experiments/baseline_lift_ppo`

运行：

```bash
cd /home/mfj/robosuite/my_experiments/baseline_lift_ppo
source /home/mfj/mj_robot/bin/activate
python lift_rl_training.py
```

## 文件整理说明

- `03_robot_comparision.py` 已重命名为 `tutorials/03_robot_comparison.py`
- `test_model_opp_date.py` 已重命名为 `baseline_lift_ppo/test_model_latest.py`
- 根目录重复的 `check_success_0323.py` 和 `debug_obs_0323.py` 已移到 `archive/duplicate_root_copies/`
- 每套实验内部的日志、模型、检查点都保留在各自目录中，避免脚本相对路径失效
- `2026-03-23_render_training/` 和 `2026-03-23_stable_training/` 的历史训练产物已经清理，只保留脚本、README 和空的 `training_runs/` 骨架
- 新的训练 run 会先写到 `training_runs/active/<run_name>/`，结束后自动归档到 `completed / interrupted / failed`
- 每个 run 都会自动生成 `run_info.json`；同级的 `training_runs/runs_index.json` 可以快速查看所有 run 的状态
- 文档里如果看到 `<timestamp>`，那只是示例占位符；真正启动 TensorBoard 前，先用 `find ./training_runs/...` 列出真实 run 名再复制路径

## 使用建议

- 想学习 robosuite 基本结构：先看 `tutorials/`
- 想看你最早怎么训练 Lift：看 `baseline_lift_ppo/`
- 想做“先靠近、再局部抓取”的新方案：看 `experiment_versions/stage2_local_grasp_v1/`
- 想继续在相对稳定的版本上训练：看 `experiment_versions/2026-03-23_stable_training/`
- 想用带渲染的老版本观察训练：看 `experiment_versions/2026-03-23_render_training/`

## 旧文档路径说明

之前一些 Markdown 文档里写的命令还是旧路径。现在如果文档里写的是：

```bash
cd /home/mfj/robosuite/my_experiments/0323_改进训练
```

请改成：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
```

如果文档里写的是：

```bash
cd /home/mfj/robosuite/my_experiments/0323_实时渲染训练
```

请改成：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_render_training
```
