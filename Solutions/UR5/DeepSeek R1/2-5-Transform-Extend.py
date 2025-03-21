import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    rx, ry, rz = r
    R_roll = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
    R_pitch = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
    R_yaw = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
    R_target = R_yaw @ R_pitch @ R_roll
    R_tcp_local_inv = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    R_total = R_target @ R_tcp_local_inv
    r13 = R_total[0, 2]
    r33 = R_total[2, 2]
    theta_sum = np.arctan2(r13, r33)
    cos_theta_sum = np.cos(theta_sum)
    sin_theta_sum = np.sin(theta_sum)
    R_Y_theta_sum_T = np.array([[cos_theta_sum, 0, -sin_theta_sum], [0, 1, 0], [sin_theta_sum, 0, cos_theta_sum]])
    N = R_Y_theta_sum_T @ R_total
    theta4 = np.arctan2(-N[2, 0], N[2, 2])
    theta3 = np.arctan2(-N[0, 1], N[1, 1])
    dx = 0.0823 * np.sin(theta4) * np.cos(theta3)
    dz_joint4 = 0.09465 + 0.0823 * np.cos(theta4)
    y_contribution = 0.093 + 0.0823 * np.sin(theta4) * np.sin(theta3)
    if not np.isclose(y_contribution, y_target, atol=0.0001):
        theta3 = -theta3
        dx = 0.0823 * np.sin(theta4) * np.cos(theta3)
        y_contribution = 0.093 + 0.0823 * np.sin(theta4) * np.sin(theta3)
        if not np.isclose(y_contribution, y_target, atol=0.0001):
            theta4 = -theta4
            dx = 0.0823 * np.sin(theta4) * np.cos(theta3)
            dz_joint4 = 0.09465 + 0.0823 * np.cos(theta4)
            y_contribution = 0.093 + 0.0823 * np.sin(theta4) * np.sin(theta3)
    A = dz_joint4 * np.sin(theta2) + dx * np.cos(theta2)
    B = dz_joint4 * np.cos(theta2) - dx * np.sin(theta2) + 0.39225
    theta1 = np.arctan2(x_target, z_target) - np.arctan2(A, B)
    magnitude = np.sqrt(A ** 2 + B ** 2)
    if not np.isclose(np.sqrt(x_target ** 2 + z_target ** 2), magnitude, atol=0.0001):
        theta1 += np.pi
        theta1 = np.arctan2(np.sin(theta1), np.cos(theta1))
    theta2 = theta_sum - theta1
    return (theta1, theta2, theta3, theta4)