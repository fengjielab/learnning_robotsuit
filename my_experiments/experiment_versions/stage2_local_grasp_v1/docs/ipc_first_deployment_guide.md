# 工控机首次部署指南

## 目标

这份文档给的是 `stage2_local_grasp_v1` 在工控机上的第一版部署方案。

目标不是一步到位做成“全自动真机抓取系统”，而是先把最关键的链路稳定跑通：

1. 工控机上能把项目环境装起来
2. 工控机上能正常加载 `A(scripted) -> B -> C` 流水线
3. 工控机上能稳定找到模型、视觉输入和相机配置
4. 先完成“可运行、可排错、可演示”的第一版

## 先说一个关键判断

你提到的 [run_staged_grasp_pipeline.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/run_staged_grasp_pipeline.py#L124) 第 `124` 行：

- `create_scripted_a_entry(...)`

它的职责是：

- 创建 `Stage A`
- 执行 scripted top-down descend
- 成功后把状态切给 `Stage B`

它不是“工控机部署入口”，而是“流水线运行时的 `A` 阶段入场逻辑”。

这点很重要，因为它决定了你的第一次部署应该怎么拆：

- 第一层：把 Python / MuJoCo / 模型 / 输入文件在工控机上跑通
- 第二层：再把这条流水线接到你的真机执行层

## 当前代码的真实边界

当前这套 `stage2_local_grasp_v1` 更准确地说是：

- 一套已经按真机流程组织好的 staged grasp pipeline
- 在 `robosuite + MuJoCo` 环境里运行
- 视觉当前只负责物体类别映射，不直接参与低层控制

当前代码里已经明确有：

- `A(scripted) -> B -> C` 流程
- `object_profile + impedance_template + stage_targets`
- 手动接管入口
- 模型与视觉输入的固定入口

当前代码里还没有直接看到：

- 你的工控机机械臂驱动接口
- PLC / 控制卡 / EtherCAT / 厂商 SDK 下发层
- 真机安全互锁、急停联动、上电使能逻辑

所以第一次部署建议不要理解成：

- “把这个脚本拷过去就能直接抓真机”

而应理解成：

- “先把工控机上的策略与流程层部署好，再接现场控制层”

## 推荐的第一次部署架构

第一版建议分成两部分：

### 部分 A：工控机上的流程层

职责：

- 读取视觉输入
- 加载 `B / C` 模型
- 执行 `A(scripted) -> B -> C`
- 输出当前阶段、目标、诊断信息

对应代码主线：

- [pipelines/run_staged_grasp_pipeline.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/run_staged_grasp_pipeline.py)
- [tools/teleop_to_staged_grasp_pipeline.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/tools/teleop_to_staged_grasp_pipeline.py)

### 部分 B：工控机上的真机执行层

职责：

- 接机器人厂商 SDK 或你现场控制程序
- 把“阶段动作”翻译成真机可执行命令
- 做限位、急停、碰撞、超时、失败回退

第一版建议人工参与：

1. 人工把夹爪送到物块正上方
2. 人工确认接管
3. 系统执行 `A(scripted) -> B -> C`

这也是当前项目已经明确推荐的主线：

- [manual_handoff_real_robot_flow.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/docs/manual_handoff_real_robot_flow.md)

## 第一次部署建议分三步走

### Step 1：先做“纯环境部署”

目标：

- 工控机上能激活 Python 环境
- 能导入 `robosuite`
- 能找到模型文件
- 能加载视觉输入 JSON

推荐做法：

1. 把整个 `robosuite` 仓库同步到工控机
2. 不要只复制单个脚本
3. 尽量保留完整目录结构

原因：

- `run_staged_grasp_pipeline.py` 依赖 `core/`
- 依赖 `training_runs/current/`
- 依赖 `training_runs/completed/`
- 依赖 `examples/`
- 依赖 `robosuite` 包本身

尤其 `training_runs/current/` 里当前使用的是软链接入口：

- [training_runs/current/README.md](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/README.md)

如果你只拷一部分文件，第一次部署时最容易踩的坑就是：

- 模型路径断掉
- 软链接失效
- 相对目录结构不一致

### Step 2：先做“仿真链路验证”

目标：

- 在工控机上先验证 `A(scripted) -> B -> C` 能跑
- 先验证软件链路，不急着第一天就接真机

建议先跑固定入口：

```bash
cd /你的路径/robosuite
python3 -m venv mj_robot
source mj_robot/bin/activate
pip install --upgrade pip
pip install -e .
pip install -r requirements-extra.txt
```

然后运行：

```bash
cd /你的路径/robosuite
source mj_robot/bin/activate
python my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/run_staged_grasp_pipeline.py \
  --vision-input /你的路径/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/vision_input_current.json \
  --camera-profile realsense_d435i \
  --stage-b-model /你的路径/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_b_model_current.zip \
  --stage-c-model /你的路径/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_c_model_current.zip \
  --episodes 1 \
  --sleep 0.03 \
  --stage-pause 1.5
```

如果工控机没有图形界面，第一次建议先加：

```bash
--no-render
```

这样做的目的不是演示效果，而是先回答三个问题：

1. Python 环境是否完整
2. 模型文件是否能正常加载
3. 运行入口是否都能找到

### Step 3：再做“真机接管版验证”

如果你的第一版现场流程是：

- 人工把夹爪送到物块正上方
- 再让系统完成抓取

那么优先使用：

- [tools/teleop_to_staged_grasp_pipeline.py](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/tools/teleop_to_staged_grasp_pipeline.py)

命令：

```bash
cd /你的路径/robosuite
source mj_robot/bin/activate
python my_experiments/experiment_versions/stage2_local_grasp_v1/tools/teleop_to_staged_grasp_pipeline.py \
  --vision-input /你的路径/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/vision_input_current.json \
  --camera-profile realsense_d435i \
  --stage-b-model /你的路径/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_b_model_current.zip \
  --stage-c-model /你的路径/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_c_model_current.zip \
  --episodes 1 \
  --sleep 0.03 \
  --stage-pause 1.5 \
  --device keyboard
```

当前建议的真机优先流程是：

1. 人工先把末端送近
2. 人工确认位置合适
3. 人工按 `p`
4. 系统进入 `A(scripted) -> B -> C`

## 我建议你第一次部署时的目录组织

如果工控机上准备长期维护，建议用下面这个结构：

```text
/opt/grasp_system/
  robosuite/
  venv/
  logs/
  runtime_inputs/
    vision_input_current.json
  launch/
    run_pipeline.sh
    run_manual_handoff.sh
```

其中：

- `robosuite/` 放项目代码
- `venv/` 放 Python 环境
- `logs/` 放运行日志
- `runtime_inputs/` 放现场替换的视觉输入
- `launch/` 放现场人员直接执行的启动脚本

## 推荐的上线脚本

第一次部署时，最好不要让现场人员手打一长串命令。

建议你在工控机上准备两个 shell 脚本。

### 1. 自动流水线脚本

文件名建议：

- `launch/run_pipeline.sh`

示例：

```bash
#!/usr/bin/env bash
set -e

BASE_DIR=/opt/grasp_system
REPO_DIR=$BASE_DIR/robosuite
VENV_DIR=$BASE_DIR/venv
VISION_INPUT=$BASE_DIR/runtime_inputs/vision_input_current.json

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

python my_experiments/experiment_versions/stage2_local_grasp_v1/pipelines/run_staged_grasp_pipeline.py \
  --vision-input "$VISION_INPUT" \
  --camera-profile realsense_d435i \
  --stage-b-model "$REPO_DIR/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_b_model_current.zip" \
  --stage-c-model "$REPO_DIR/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_c_model_current.zip" \
  --episodes 1 \
  --sleep 0.03 \
  --stage-pause 1.5
```

### 2. 人工接管脚本

文件名建议：

- `launch/run_manual_handoff.sh`

示例：

```bash
#!/usr/bin/env bash
set -e

BASE_DIR=/opt/grasp_system
REPO_DIR=$BASE_DIR/robosuite
VENV_DIR=$BASE_DIR/venv
VISION_INPUT=$BASE_DIR/runtime_inputs/vision_input_current.json

source "$VENV_DIR/bin/activate"
cd "$REPO_DIR"

python my_experiments/experiment_versions/stage2_local_grasp_v1/tools/teleop_to_staged_grasp_pipeline.py \
  --vision-input "$VISION_INPUT" \
  --camera-profile realsense_d435i \
  --stage-b-model "$REPO_DIR/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_b_model_current.zip" \
  --stage-c-model "$REPO_DIR/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_c_model_current.zip" \
  --episodes 1 \
  --sleep 0.03 \
  --stage-pause 1.5 \
  --device keyboard
```

## 第一次部署时最推荐的输入文件策略

当前项目里已经有固定入口：

- [training_runs/current/stage_b_model_current.zip](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_b_model_current.zip)
- [training_runs/current/stage_c_model_current.zip](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/stage_c_model_current.zip)
- [training_runs/current/vision_input_current.json](/home/mfj/robosuite/my_experiments/experiment_versions/stage2_local_grasp_v1/training_runs/current/vision_input_current.json)

第一次部署建议：

- 模型先固定，不要一开始就频繁切换
- `vision_input_current.json` 作为现场统一入口
- 真正的视觉模块输出只需要覆盖这个 JSON

当前 JSON 最少需要这些字段：

```json
{
  "object_class": "cube",
  "shape_tag": "box",
  "size_tag": "small",
  "object_profile_name": "cube_small",
  "detection_confidence": 0.98,
  "source": "realsense_d435i_demo"
}
```

## 第一次上工控机的推荐验收顺序

不要一上来就做全链路真机验证。

建议按下面顺序验收：

### 验收 1：环境验收

检查：

- `python -c "import robosuite"`
- `python -c "import mujoco"`
- `python -c "import pynput"`

目标：

- 先确认环境包没缺

### 验收 2：文件验收

检查：

- `stage_b_model_current.zip` 是否存在
- `stage_c_model_current.zip` 是否存在
- `vision_input_current.json` 是否存在

目标：

- 先确认路径没断

### 验收 3：流水线验收

运行：

- `run_staged_grasp_pipeline.py`

目标：

- 看脚本能不能完整进入 `A(scripted) -> B -> C`

### 验收 4：人工接管验收

运行：

- `teleop_to_staged_grasp_pipeline.py`

目标：

- 看人工接管流程能不能正常开始

### 验收 5：真机桥接验收

这一步才轮到你自己的机器人接口层。

目标：

- 把 staged pipeline 的阶段动作真正发到真机
- 做速度、位移、力、超时和急停限制

## 第一次部署时的几个高优先级风险

### 风险 1：误以为脚本已经直接控制真机

当前不是。

当前更接近：

- 任务流程层
- 仿真验证层
- 策略组织层

如果你要直接下发真机，需要补你自己的机器人适配层。

### 风险 2：工控机只拷了脚本，没有拷模型和目录结构

这个最常见。

第一次部署尽量同步整个仓库，至少保留：

- `robosuite/`
- `my_experiments/experiment_versions/stage2_local_grasp_v1/`
- `training_runs/current/`
- `training_runs/completed/`

### 风险 3：第一次就想做自动接管

当前主线更推荐手动接管。

原因：

- 真机第一版更安全
- 更容易排查到底是“送近没到位”，还是 `B/C` 策略问题

### 风险 4：渲染或图形环境卡住部署

如果工控机是无桌面或远程桌面环境，第一次先用：

- `--no-render`

先把主链路跑通，再处理显示问题。

## 最推荐的首次上线版本

如果你现在就要做第一版，我建议用下面这个组合：

1. 工控机部署整个 `robosuite` 项目
2. 建独立 venv
3. 固定使用 `training_runs/current/` 作为模型入口
4. 固定使用 `vision_input_current.json` 作为视觉入口
5. 现场优先用“人工接管版”
6. 暂时不做自动接管
7. 暂时不直接把策略输出裸连真机
8. 先补一层真机安全执行层再真正闭环

一句话总结：

第一次部署最稳的方式不是“直接把脚本扔到工控机上跑真机”，而是“先把工控机上的流程层跑稳，再通过人工接管和安全执行层接入真机”。
