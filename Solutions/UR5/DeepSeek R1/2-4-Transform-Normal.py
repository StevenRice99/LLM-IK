import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    x_target, y_target, z_target = p
    rx, ry, rz = r
    R_roll = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    R_pitch = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R_yaw = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_target = R_yaw @ R_pitch @ R_roll
    r13 = R_target[0, 2]
    r33 = R_target[2, 2]
    theta_sum = np.arctan2(r13, r33)
    r21 = R_target[1, 0]
    r22 = R_target[1, 1]
    theta3 = np.arctan2(r21, r22)
    A = 0.39225
    B = 0.09465
    sin_S = np.sin(theta_sum)
    cos_S = np.cos(theta_sum)
    numerator_sin = x_target - B * sin_S
    numerator_cos = z_target - B * cos_S
    theta1 = np.arctan2(numerator_sin, numerator_cos)
    theta2 = theta_sum - theta1
    return (theta1, theta2, theta3)