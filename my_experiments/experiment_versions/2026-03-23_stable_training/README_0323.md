# 2026-03-23_stable_training - 文件整理和说明

**创建日期**：2026 年 3 月 23 日

**目的**：整理之前的训练文件，并创建改进版训练脚本

---

## 📁 文件夹结构

```
2026-03-23_stable_training/
├── README_0323.md              # 本说明文件
├── lift_rl_training_原版_0318.py    # 原始训练脚本（3 月 18 日）
├── continue_training_0320.py   # 继续训练脚本（3 月 20 日）
├── test_model_0323.py          # 测试模型脚本（3 月 23 日）
├── check_success_0323.py       # 检查成功标准脚本（3 月 23 日）
├── debug_obs_0323.py           # 调试观测值脚本（3 月 23 日）
├── plot_training_results.py    # 绘图脚本
│
└── lift_rl_training_v2_0323.py  # 改进版训练脚本（新建）
```

---

## 📅 训练历史

### 3 月 18 日 - 第一次训练
- **文件**：`lift_rl_training_原版_0318.py`
- **训练步数**：10 万步
- **模型**：`ppo_lift_final.zip`
- **结果**：模型学会了接近方块，但抓取和举起不稳定

### 3 月 20 日 - 继续训练
- **文件**：`continue_training_0320.py`
- **训练步数**：从 20 万步到 60 万步
- **模型**：`ppo_lift_final_v2.zip` → `ppo_lift_final_v3.zip`
- **结果**：有所改善，但仍然不能稳定抓取

### 3 月 23 日 - 问题分析和改进
- **发现问题**：
  - 平均最大相对高度只有 0.0022 米
  - 成功标准需要 0.04 米
  - 模型只学会了接近方块，没学会抓取和举起

- **原因分析**：
  1. 训练步数不够（60 万步可能不够）
  2. 奖励函数分配不合理
  3. 超参数可能需要调整

- **解决方案**：创建改进版训练脚本 `lift_rl_training_v2_0323.py`

---

## 🚀 改进版训练脚本功能

### 1. 自定义奖励函数
- 增加抓取成功的奖励
- 增加举起高度的奖励权重
- 减少接近奖励的权重

### 2. 更多训练步数
- 目标：100 万步
- 检查点：每 10 万步保存一次

### 3. 更好的超参数
- 增加探索率（ent_coef）
- 调整学习率
- 增加 n_epochs

### 4. 课程学习（可选）
- 第一阶段：学会接近方块
- 第二阶段：学会抓取方块
- 第三阶段：学会举起方块

---

## 📊 模型文件位置

这些较早的模型和检查点现在集中放在：

```text
/home/mfj/robosuite/my_experiments/baseline_lift_ppo/
```

---

## 🎯 下一步操作

### 1. 运行改进版训练
```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
python lift_rl_training_v2_0323.py
```

### 2. 监控训练
```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
find ./training_runs/active ./training_runs/completed ./training_runs/interrupted ./training_runs/failed -maxdepth 1 -mindepth 1 -type d | sort
tensorboard --logdir ./training_runs/completed/lift_ppo_v3_20260406_103000/tensorboard
```

### 3. 测试模型
```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
python test_model_0323.py
```

### 4. 检查成功标准
```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
python check_success_0323.py
```

---

## 📝 文件命名规范

从今以后，所有文件都使用以下格式：
```
文件名_YYYYMMDD.py
```

例如：
- `lift_rl_training_v2_0323.py` - 3 月 23 日创建的改进版训练脚本
- `test_model_0323.py` - 3 月 23 日创建的测试脚本

---

**祝训练顺利！🎉**
