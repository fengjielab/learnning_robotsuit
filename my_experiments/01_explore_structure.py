import os
import robosuite as suite

print("=" * 60)
print("阶段 1：robosuite 内部结构解剖")
print("=" * 60)

# 1. 看安装路径
print(f"\n📁 robosuite 安装位置：{suite.__file__}")
base_path = os.path.dirname(suite.__file__)

# 2. 遍历关键目录
folders = ['controllers', 'environments', 'models', 'robots', 'demos']
for folder in folders:
    path = os.path.join(base_path, folder)
    files = os.listdir(path)[:5]  # 只看前5个文件
    print(f"\n📂 {folder}/")
    print(f"   里面有什么：{', '.join(files)}...")
    print(f"   文件数：{len(os.listdir(path))}")

# 3. 看可用的环境、机器人、控制器
print("\n" + "=" * 60)
print("可用资源清单：")
print("=" * 60)
print(f"环境数：{len(suite.ALL_ENVIRONMENTS)}")
print(f"前10个环境：{suite.ALL_ENVIRONMENTS[:10]}")
print(f"\n机器人类：{suite.ALL_ROBOTS}")
print(f"\n控制器类型：{suite.ALL_CONTROLLERS}")