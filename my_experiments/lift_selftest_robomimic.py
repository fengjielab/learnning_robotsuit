def test_pretrained():
    print("\n" + "=" * 70)
    print("方法 4：预训练神经网络")
    print("=" * 70)
    
    print("""
需要先安装 robomimic 并下载模型：

pip install robomimic

# 下载预训练模型
wget https://robomimic.github.io/models/datasets/lift_ph_can_bc.pth

然后运行以下代码：

```python
import torch
from robomimic.algo.bc import BC

# 加载模型
model = BC.load("lift_ph_can_bc.pth")

# 使用模型
obs = env.reset()
for i in range(200):
    # 转换观测值为 tensor
    obs_tensor = {k: torch.FloatTensor(v).unsqueeze(0) for k, v in obs.items()}
    # 预测动作
    action = model.predict(obs_tensor, deterministic=True)
    # 执行
    obs, reward, done, info = env.step(action.numpy()[0])
```

评价：效果最好，但需要额外依赖
""")