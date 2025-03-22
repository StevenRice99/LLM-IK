import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Computes the 5-DOF closed-form inverse kinematics solution for the serial manipulator.
    The joints correspond to:
      - Revolute 1 (axis Y)
      - Revolute 2 (axis Y)
      - Revolute 3 (axis Y)
      - Revolute 4 (axis Z)
      - Revolute 5 (axis Y)
    The robot has a fixed tool offset with translation [0, 0.0823, 0] and an extra rotation of R_z(1.570796325)
    (so that when all joint angles are 0, the TCP pose is 
     Position: [0, 0.0556, 0.9119] with Orientation: [0, 0, 1.570796325]).
    
    :param p: The desired TCP position as (x, y, z)
    :param r: The desired TCP orientation in roll, pitch, yaw (in radians)
    :return: A tuple (θ₁, θ₂, θ₃, θ₄, θ₅) representing the joint angles in radians.
    """
    x_target, y_target, z_target = p
    roll, pitch, yaw = r
    cr, sr = (math.cos(roll), math.sin(roll))
    cp, sp = (math.cos(pitch), math.sin(pitch))
    cy, sy = (math.cos(yaw), math.sin(yaw))
    R_x = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    R_y = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    R_z = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    R_target = R_z @ R_y @ R_x
    yaw_tcp = 1.570796325
    R_tcp_fixed = np.array([[math.cos(yaw_tcp), -math.sin(yaw_tcp), 0], [math.sin(yaw_tcp), math.cos(yaw_tcp), 0], [0, 0, 1]])
    p_tcp = np.array([x_target, y_target, z_target])
    p_wrist = p_tcp - np.array([0, 0.0823, 0])
    theta_sum = math.atan2(R_target[0, 2], R_target[2, 2])
    C_tcp = 0.09465
    x_wrist = p_wrist[0]
    z_wrist = p_wrist[2]
    x_eff = x_wrist - C_tcp * math.sin(theta_sum)
    z_eff = z_wrist - C_tcp * math.cos(theta_sum)
    a = 0.425
    b = 0.39225
    d_sq = x_eff ** 2 + z_eff ** 2
    cos_theta2 = (d_sq - a ** 2 - b ** 2) / (2 * a * b)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(x_eff, z_eff) - math.atan2(b * math.sin(theta2), a + b * math.cos(theta2))
    theta3 = theta_sum - theta1 - theta2
    M = R_target @ np.linalg.inv(R_tcp_fixed)
    R_y_sum = np.array([[math.cos(theta_sum), 0, math.sin(theta_sum)], [0, 1, 0], [-math.sin(theta_sum), 0, math.cos(theta_sum)]])
    R_bar = R_y_sum.T @ M
    theta5 = -math.asin(R_bar[2, 0])
    theta4 = math.atan2(R_bar[1, 0], R_bar[0, 0])
    return (theta1, theta2, theta3, theta4, theta5)