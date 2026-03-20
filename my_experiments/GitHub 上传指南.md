# 将你的代码上传到 GitHub 完整指南

## 概述

本指南将帮助你把本地的修改和实验文件上传到你自己的 GitHub 仓库，这样即使换电脑也不会丢失代码。

---

## 第一步：确认你的 GitHub 仓库

1. 打开浏览器，访问 https://github.com
2. 登录你的 GitHub 账号
3. 确认你已经创建了一个新仓库（例如：`robosuite-my-experiments` 或 `robosuite-learning`）
4. 复制仓库的 URL，格式类似：`https://github.com/你的用户名/仓库名.git`

---

## 第二步：更新 .gitignore 文件（可选但推荐）

你创建的训练数据文件（检查点、日志、图片等）很大，不需要上传到 GitHub。

编辑 `robosuite/.gitignore` 文件，在末尾添加以下内容：

```
# 实验数据（训练产生的大文件）
my_experiments/ppo_checkpoints/
my_experiments/ppo_logs/
my_experiments/ppo_tensorboard/
my_experiments/*.zip
my_experiments/*.png
my_experiments/*.json
```

保存文件。

---

## 第三步：查看当前 Git 状态

打开终端，进入 robosuite 目录，运行：

```bash
cd /home/mfj/robosuite
git status
```

你会看到类似这样的输出：
- **未跟踪的文件**：表示 Git 还没开始跟踪这些新文件
- **已修改的文件**：表示你修改了已有的文件

---

## 第四步：添加文件到 Git

将所有修改和新文件添加到 Git 暂存区：

```bash
git add .
```

或者只添加你创建的文件：

```bash
git add my_experiments/
git add LEARNING_GUIDE.md
git add docs/modules/environments_zh.md
git add docs/modules/overview_zh.md
```

---

## 第五步：提交更改

提交你的更改到本地仓库：

```bash
git commit -m "添加我的实验文件和学习指南"
```

你可以自定义提交信息，例如：
```bash
git commit -m "feat: 添加 PPO 训练实验和完整学习指南"
```

---

## 第六步：配置远程仓库

因为当前仓库已经有官方的远程地址，你需要添加你自己的远程仓库。

**方法 A：添加新的远程地址（推荐）**

```bash
git remote add myrepo https://github.com/你的用户名/你的仓库名.git
```

将 `你的用户名` 和 `你的仓库名` 替换成实际的 GitHub 用户名和仓库名。

**方法 B：修改现有的远程地址**

如果你想完全替换远程地址：

```bash
git remote set-url origin https://github.com/你的用户名/你的仓库名.git
```

---

## 第七步：推送到 GitHub

将代码推送到你的 GitHub 仓库：

```bash
git push -u myrepo main
```

或者如果使用 origin：

```bash
git push -u origin main
```

如果是第一次推送，可能需要输入 GitHub 用户名和密码（或使用 Personal Access Token）。

---

## 第八步：验证上传成功

1. 打开浏览器，访问你的 GitHub 仓库
2. 刷新页面，你应该能看到新上传的文件
3. 检查 `my_experiments/` 目录和你的文档文件是否都在

---

## 常见问题

### 问题 1：推送失败，提示 "remote origin already exists"

**解决方法**：使用不同的远程名称
```bash
git remote add mygithub https://github.com/你的用户名/仓库名.git
git push -u mygithub main
```

### 问题 2：推送失败，提示需要认证

**解决方法**：使用 Personal Access Token
1. 访问 https://github.com/settings/tokens
2. 创建一个新的 Token（勾选 `repo` 权限）
3. 推送时使用 Token 作为密码

### 问题 3：文件太大被拒绝

**解决方法**：检查 .gitignore 是否正确配置，然后移除大文件
```bash
git rm --cached my_experiments/ppo_checkpoints/*.zip
git commit -m "移除大文件"
git push
```

### 问题 4：分支名称不是 main

**解决方法**：有些仓库使用 master 作为主分支
```bash
git push -u origin master
```

查看当前分支名：
```bash
git branch
```

---

## 换电脑后如何下载代码

在新电脑上，只需运行：

```bash
git clone https://github.com/你的用户名/你的仓库名.git
```

然后安装依赖：
```bash
cd 你的仓库名
pip install -e .
```

---

## 日常使用流程

每次完成工作后，保存代码到 GitHub：

```bash
# 1. 查看更改
git status

# 2. 添加更改
git add .

# 3. 提交更改
git commit -m "描述你做了什么"

# 4. 推送到 GitHub
git push
```

---

## 总结

完成以上步骤后，你的所有代码和实验文件都会安全地保存在 GitHub 上。无论换多少台电脑，都可以随时下载回来！

**核心命令速查**：
```bash
git add .                    # 添加所有更改
git commit -m "提交信息"      # 提交到本地
git push                     # 推送到 GitHub
```

祝你使用愉快！