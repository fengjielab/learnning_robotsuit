# 2026-03-23_stable_training - 完整指南

**创建日期**：2026 年 3 月 23 日  
**最后更新**：2026 年 3 月 23 日

---

## 📁 目录结构

```
2026-03-23_stable_training/
│
├── README_完整指南_0323.md       ← 你正在看的文件
├── README_0323.md                ← 旧版说明（保留参考）
│
├── 训练脚本/
│   ├── lift_rl_training_v2_0323.py    ← v2 训练脚本（奖励改进，但不稳定）
│   ├── lift_rl_training_v3_0323.py    ← v3 训练脚本（稳定版）
│   └── continue_training_v3_0323.py   ← 继续训练脚本（改进超参数）
│
├── 测试脚本/
│   ├── test_model_0323.py             ← 测试模型
│   ├── check_success_0323.py          ← 检查成功标准
│   └── debug_obs_0323.py              ← 调试观测值
│
├── 工具脚本/
│   └── plot_training_results.py       ← 绘制训练曲线
│
├── 参考脚本/
│   ├── lift_rl_training_原版_0318.py  ← 原始训练脚本（3 月 18 日）
│   └── continue_training_0320.py      ← 旧版继续训练（3 月 20 日）
│
└── training_runs/
    ├── active/                        ← 正在运行的实验
    ├── completed/                     ← 正常结束的实验
    ├── interrupted/                   ← 手动中断但可恢复的实验
    ├── failed/                        ← 失败的实验
    └── runs_index.json                ← 所有 run 的索引
```

---

## 📅 训练历史

### 3 月 18 日 - 第一次训练
| 项目 | 值 |
|------|-----|
| 脚本 | `lift_rl_training_原版_0318.py` |
| 步数 | 10 万步 |
| 模型 | `ppo_lift_final.zip` |
| 结果 | 学会接近方块，抓取不稳定 |

### 3 月 20 日 - 继续训练
| 项目 | 值 |
|------|-----|
| 脚本 | `continue_training_0320.py` |
| 步数 | 20 万 → 60 万步 |
| 模型 | `ppo_lift_final_v3.zip` |
| 结果 | 有改善，但仍不能稳定抓取 |

### 3 月 23 日 - 问题分析
**发现的问题**：
- 平均最大相对高度：0.0022 米
- 成功标准：0.04 米
- 模型只学会接近，没学会抓取和举起

**原因分析**：
1. 默认奖励函数不合理
2. 训练步数不够
3. 超参数需要调整

### 3 月 23 日 - v2 训练脚本
| 项目 | 值 |
|------|-----|
| 脚本 | `lift_rl_training_v2_0323.py` |
| 改进 | 自定义奖励函数 |
| 超参数 | ent_coef=0.03, lr=3e-4 |
| 结果 | ❌ 训练不稳定，std 飙升到 20+ |

### 3 月 23 日 - v3 训练脚本（稳定版）
| 项目 | 值 |
|------|-----|
| 脚本 | `lift_rl_training_v3_0323.py` |
| 改进 | 降低探索率和学习率 |
| 超参数 | ent_coef=0.01, lr=1e-4, clip=0.1 |
| 结果 | ✅ 训练稳定，std=1.1 |

### 3 月 23 日 - 继续训练（改进超参数）
| 项目 | 值 |
|------|-----|
| 脚本 | `continue_training_v3_0323.py` |
| 改进 | 进一步降低学习率，增大 batch |
| 超参数 | lr=5e-5, batch=128, epochs=15 |
| 状态 | 🔄 进行中 |

---

## 🚀 快速开始

### 方案 A：从头开始训练（不推荐，浪费时间）
```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
source /home/mfj/mj_robot/bin/activate
python lift_rl_training_v3_0323.py
```

### 方案 B：从检查点继续训练（推荐）
```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
source /home/mfj/mj_robot/bin/activate
python continue_training_v3_0323.py
```

---

## 📊 超参数对比

| 参数 | v2（不稳定） | v3（稳定） | 继续训练 |
|------|-------------|-----------|----------|
| learning_rate | 3e-4 | 1e-4 | 5e-5 |
| batch_size | 128 | 64 | 128 |
| n_epochs | 20 | 10 | 15 |
| ent_coef | 0.03 | 0.01 | 0.01 |
| clip_range | 0.2 | 0.1 | 0.1 |
| max_grad_norm | 0.5 | 0.3 | 0.3 |

---

## 📁 文件说明

### 训练脚本

| 文件 | 用途 | 状态 |
|------|------|------|
| `lift_rl_training_v2_0323.py` | v2 训练（自定义奖励） | ❌ 已废弃（不稳定） |
| `lift_rl_training_v3_0323.py` | v3 训练（稳定版） | ✅ 可用 |
| `continue_training_v3_0323.py` | 从检查点继续训练 | ✅ 推荐使用 |

### 测试脚本

| 文件 | 用途 | 命令 |
|------|------|------|
| `test_model_0323.py` | 测试模型表现 | `python test_model_0323.py` |
| `check_success_0323.py` | 检查成功标准 | `python check_success_0323.py` |
| `debug_obs_0323.py` | 调试观测值结构 | `python debug_obs_0323.py` |

### 生成的配置文件

| 文件 | 说明 |
|------|------|
| `ppo_training_config_v3_0323.json` | v3 训练配置 |
| `ppo_training_config_continue_v6_*.json` | 继续训练配置 |

---

## 📈 TensorBoard 监控

### 启动 TensorBoard
训练脚本现在会为每次运行自动创建独立 run，并按状态归档到：
- `training_runs/active/`
- `training_runs/completed/`
- `training_runs/interrupted/`
- `training_runs/failed/`

因此请优先使用终端打印出来的那次运行专属路径启动 TensorBoard。

先列出真实 run 名：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
find ./training_runs/active ./training_runs/completed ./training_runs/interrupted ./training_runs/failed -maxdepth 1 -mindepth 1 -type d | sort
```

然后再复制真实目录启动 TensorBoard。示例：

```bash
cd /home/mfj/robosuite/my_experiments/experiment_versions/2026-03-23_stable_training
source /home/mfj/mj_robot/bin/activate
tensorboard --logdir ./training_runs/completed/lift_ppo_v3_20260406_103000/tensorboard --host 0.0.0.0 --port 6007
```

### 浏览器访问
```
http://127.0.0.1:6007
```

### 关键指标
| 指标 | 正常范围 | 说明 |
|------|----------|------|
| `rollout/ep_rew_mean` | 逐渐上升 | 平均奖励，应该增加 |
| `train/value_loss` | 稳定或下降 | 价值损失，不应爆炸 |
| `train/std` | 0.5-3 | 策略标准差，应稳定 |
| `train/clip_fraction` | 0.05-0.2 | 裁剪比例 |
| `train/explained_variance` | > 0 | 越高越好 |

---

## 🎯 成功标准

### Lift 任务成功条件
- **方块相对桌面高度 > 0.04 米**
- 桌面高度：0.83 米（世界坐标）
- 成功高度：0.87 米（世界坐标）

### 评估脚本
```bash
python check_success_0323.py
```

### 输出说明
- 最大相对高度 > 0.04 米 → 成功
- 成功率 > 80% → 模型优秀
- 成功率 > 50% → 模型良好

---

## 🔧 常见问题

### Q1: TensorBoard 不显示数据
**解决**：
1. 确认训练正在运行
2. 等 30-60 秒让数据写入
3. 刷新浏览器（Ctrl+Shift+R）
4. 确认日志目录正确

### Q2: 训练中断后如何继续
**解决**：
```bash
python continue_training_v3_0323.py
```

### Q3: value_loss 越来越大
**解决**：
1. 降低学习率
2. 增大 batch size
3. 增加 n_epochs
4. 简化奖励函数

### Q4: 模型不学习（奖励不上升）
**解决**：
1. 检查奖励函数
2. 增加训练步数
3. 调整探索率

---

## 📋 检查清单

### 训练前
- [ ] 激活虚拟环境
- [ ] 确认检查点存在
- [ ] 启动 TensorBoard

### 训练中
- [ ] 观察 `ep_rew_mean` 是否上升
- [ ] 观察 `value_loss` 是否稳定
- [ ] 观察 `std` 是否在正常范围

### 训练后
- [ ] 运行 `check_success_0323.py` 测试
- [ ] 运行 `test_model_0323.py` 可视化测试
- [ ] 保存模型和配置

---

## 📞 文件版本对应关系

| 训练版本 | 训练脚本 | 继续训练脚本 | 模型文件 | 配置文件 |
|----------|----------|--------------|----------|----------|
| v2 | `lift_rl_training_v2_0323.py` | - | `ppo_lift_final_v4_0323.zip` | `ppo_training_config_v2_0323.json` |
| v3 | `lift_rl_training_v3_0323.py` | `continue_training_v3_0323.py` | `ppo_lift_final_v5_0323.zip` | `ppo_training_config_v3_0323.json` |
| v6（继续） | - | `continue_training_v3_0323.py` | `ppo_lift_final_v6_0323_continued_*.zip` | `ppo_training_config_continue_v6_*.json` |

---

**祝训练顺利！🎉**

如有问题，参考本指南或查看 TensorBoard 监控数据。
