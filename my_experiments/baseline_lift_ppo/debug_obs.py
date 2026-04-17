"""调试：查看观测值的详细结构"""

import robosuite as suite
import numpy as np

env = suite.make(
    "Lift",
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    use_object_obs=True,
    reward_shaping=True,
    control_freq=20,
    horizon=500,
)

obs = env.reset()

print("=" * 70)
print("观测值结构")
print("=" * 70)
print(f"类型：{type(obs)}")
print(f"键名：{list(obs.keys())}")
print()

for key in obs.keys():
    val = obs[key]
    if isinstance(val, np.ndarray):
        print(f"{key}:")
        print(f"  形状：{val.shape}")
        print(f"  类型：{val.dtype}")
        print(f"  值范围：[{val.min():.4f}, {val.max():.4f}]")
        print(f"  前 10 个值：{val[:10]}")
        print()

# 尝试从环境内部获取方块位置
print("=" * 70)
print("从环境内部获取方块位置")
print("=" * 70)

if hasattr(env, 'model'):
    print(f"env.model 属性：{dir(env.model)}")
    if hasattr(env.model, 'cube_id'):
        print(f"cube_id: {env.model.cube_id}")
    if hasattr(env.model, 'object_id'):
        print(f"object_id: {env.model.object_id}")

if hasattr(env, 'sim'):
    print(f"\nenv.sim 可用")
    # 尝试获取所有物体位置
    try:
        n_bodies = env.sim.model.nbody
        print(f"总身体数：{n_bodies}")
        for i in range(min(n_bodies, 20)):  # 只显示前 20 个
            name = env.sim.model.body_name(i)
            pos = env.sim.data.body_xpos[i]
            print(f"  {i}: {name} - 位置：[{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]")
    except Exception as e:
        print(f"错误：{e}")

env.close()
