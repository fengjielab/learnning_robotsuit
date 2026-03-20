import robosuite as suite
import numpy as np

# 只选单臂环境（排除需要特殊配置的双臂环境）
envs_to_test = [
    ("Lift", "基础抓取：举起方块"),
    ("Door", "接触力任务：拉开抽屉"),
    ("PickPlaceCan", "视觉分拣：抓滑溜的罐头"),
    ("NutAssemblySquare", "精密装配：拧方螺母"),
]

print("🎬 环境大观园（修复版）")
print("每个环境跑 150 步（约 7.5 秒），观察机器人动作")
print("=" * 60)

for env_name, description in envs_to_test:
    print(f"\n📍 正在加载：{env_name}")
    print(f"   任务：{description}")
    input("   按 Enter 开始（或 Ctrl+C 跳过）...")
    
    try:
        # 关键修复 1：使用 action_dim 而不是 action_space
        env = suite.make(
            env_name=env_name,
            robots="Panda",  # 单臂
            has_renderer=True,
            has_offscreen_renderer=False,
            use_camera_obs=False,
            control_freq=20,
        )
        
        env.reset()
        
        # 关键修复 2：用 np.random.randn(env.action_dim) 生成动作
        for i in range(150):
            action = np.random.randn(env.action_dim) * 0.5  # 随机动作，幅度小一点
            obs, reward, done, info = env.step(action)
            env.render()
            
            if i % 50 == 0:
                print(f"      Step {i}: Reward={reward:.2f}, 动作维度={env.action_dim}")
                
            if done:
                print(f"      ✅ 任务完成/重置")
                break
                
        env.close()
        print(f"   ✓ {env_name} 完成")
        
    except KeyboardInterrupt:
        print(f"   ⏭️  跳过")
        try:
            env.close()
        except:
            pass
        continue
    except Exception as e:
        print(f"   ❌ 错误：{e}")
        continue

print("\n🎉 完成！你看到了：")
print("- Lift：红色方块，机器人尝试抓取")
print("- Door：抽屉/门把手，需要旋转+拉")
print("- PickPlaceCan：圆柱形罐头，容易滚动")
print("- NutAssembly：螺纹装配，需要旋转")