"""
===============================================================================
                    绘制训练结果图表
===============================================================================

这个脚本会读取训练日志并绘制：
1. 奖励曲线（每集的总奖励）
2. 奖励趋势（滑动平均）
3. 训练时长统计
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

# =============================================================================
# 读取监控数据
# =============================================================================

print("=" * 70)
print("读取训练数据")
print("=" * 70)

log_dir = "./ppo_logs"
monitor_path = os.path.join(log_dir, "monitor.csv")

if not os.path.exists(monitor_path):
    print(f"\n❌ 找不到监控文件：{monitor_path}")
    print("请先运行训练脚本生成数据！")
else:
    # 读取数据
    df = pd.read_csv(monitor_path, skiprows=1)  # 跳过注释行
    
    print(f"\n✅ 找到 {len(df)} 集训练数据")
    print(f"\n数据列：{list(df.columns)}")
    print(f"\n前 5 集数据:")
    print(df.head())
    
    # =============================================================================
    # 绘制奖励曲线
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("绘制图表")
    print("=" * 70)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图 1：原始奖励曲线
    ax1 = axes[0]
    ax1.plot(df['r'].values, alpha=0.7, label='原始奖励', color='blue')
    
    # 滑动平均（窗口大小 10）
    window = min(10, len(df) // 5)
    if window > 0:
        rolling_mean = df['r'].rolling(window=window).mean()
        ax1.plot(rolling_mean.values, label=f'{window}集滑动平均', color='red', linewidth=2)
    
    ax1.set_xlabel('集数 (Episode)')
    ax1.set_ylabel('奖励 (Reward)')
    ax1.set_title('训练奖励曲线')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 图 2：奖励分布
    ax2 = axes[1]
    ax2.hist(df['r'].values, bins=20, alpha=0.7, color='green', edgecolor='black')
    ax2.axvline(df['r'].mean(), color='red', linestyle='--', linewidth=2, label=f'平均值：{df["r"].mean():.2f}')
    ax2.set_xlabel('奖励')
    ax2.set_ylabel('频数')
    ax2.set_title('奖励分布直方图')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图片
    save_path = "./training_reward_plot.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ 图表已保存到：{save_path}")
    
    # =============================================================================
    # 打印统计信息
    # =============================================================================
    
    print("\n" + "=" * 70)
    print("训练统计")
    print("=" * 70)
    
    print(f"""
    总集数：{len(df)}
    
    奖励统计:
        最小值：{df['r'].min():.4f}
        最大值：{df['r'].max():.4f}
        平均值：{df['r'].mean():.4f}
        中位数：{df['r'].median():.4f}
        标准差：{df['r'].std():.4f}
    
    最后 10 集平均奖励：{df['r'].tail(10).mean():.4f}
    最早 10 集平均奖励：{df['r'].head(10).mean():.4f}
    
    进步：{df['r'].tail(10).mean() - df['r'].head(10).mean():.4f}
    """)
    
    # 检查是否有时间数据
    if 'l' in df.columns:
        total_time = df['l'].sum()
        print(f"""
    时长统计:
        总步数：{total_time}
        平均每集步数：{df['l'].mean():.1f}
        """)
    
    plt.show()

print("\n" + "=" * 70)
print("完成！")
print("=" * 70)