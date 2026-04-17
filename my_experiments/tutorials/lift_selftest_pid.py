import robosuite as suite
import numpy as np

class PIDController:
    def __init__(self, kp=1.0, ki=0.01, kd=0.1):
        self.kp = kp  # 比例增益
        self.ki = ki  # 积分增益
        self.kd = kd  # 微分增益
        self.integral = 0
        self.prev_error = 0
    
    def compute(self, error, dt=0.05):
        """
        计算 PID 输出
        
        error: 目标值 - 当前值
        dt: 时间间隔
        """
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        
        self.prev_error = error
        return output

def pid_policy(env, obs, pid_x, pid_y, pid_z):
    """
    使用 PID 控制器
    
    思路：用 3 个 PID 分别控制 X、Y、Z 方向的移动
    """
    gripper_pos = obs['robot0_eef_pos']
    cube_pos = obs['cube_pos']
    
    # 计算误差（目标 - 当前）
    error_x = cube_pos[0] - gripper_pos[0]
    error_y = cube_pos[1] - gripper_pos[1]
    error_z = cube_pos[2] - gripper_pos[2] + 0.1  # 目标在方块上方 10cm
    
    # 用 PID 计算每个方向的动作
    action_x = pid_x.compute(error_x)
    action_y = pid_y.compute(error_y)
    action_z = pid_z.compute(error_z)
    
    # 组合成完整动作（前 3 维是位置，最后 1 维是夹爪）
    action = np.array([action_x, action_y, action_z, 0, 0, 0, 0, -0.5])
    
    return np.clip(action, -1, 1)

def test_pid():
    print("\n" + "=" * 70)
    print("方法 3：PID 控制器")
    print("=" * 70)
    
    # 创建 3 个 PID 控制器
    pid_x = PIDController(kp=2.0, ki=0.1, kd=0.5)
    pid_y = PIDController(kp=2.0, ki=0.1, kd=0.5)
    pid_z = PIDController(kp=2.0, ki=0.1, kd=0.5)
    
    env = suite.make(
        "Lift",
        robots="Panda",
        has_renderer=True,
        reward_shaping=True,
        control_freq=20,
        horizon=2000,
    )
    
    obs = env.reset()
    total_reward = 0
    
    for i in range(2000):
        action = pid_policy(env, obs, pid_x, pid_y, pid_z)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        
        if i % 50 == 0:
            print(f"步{i}: 奖励={reward:.4f}")
    
    print(f"\nPID 控制器总奖励：{total_reward:.4f}")
    print("评价：比启发式更平滑，但参数需要调试")
    env.close()

test_pid()