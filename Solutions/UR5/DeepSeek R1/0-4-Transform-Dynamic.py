import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_tcp, y_tcp, z_tcp = p
    roll, pitch, yaw = r
    theta1 = math.atan2(y_tcp, -x_tcp)
    sin_theta1 = math.sin(theta1)
    cos_theta1 = math.cos(theta1)
    j2_x = 0.13585 * sin_theta1
    j2_y = 0.13585 * cos_theta1
    j2_z = 0.0
    dx = x_tcp - j2_x
    dy = y_tcp - j2_y
    dz = z_tcp - j2_z
    adj_x = dx * cos_theta1 + dy * sin_theta1
    adj_y = -dx * sin_theta1 + dy * cos_theta1
    adj_z = dz
    target_x = adj_x
    target_y = adj_y + 0.1197
    target_z = adj_z - 0.425
    a = 0.39225
    b = 0.093 + 0.09465
    x = math.sqrt(target_x ** 2 + target_z ** 2)
    z = target_y
    D = (x ** 2 + z ** 2 - a ** 2 - b ** 2) / (2 * a * b)
    D = max(min(D, 1.0), -1.0)
    theta3 = math.acos(D)
    solutions = []
    for theta3_sol in [theta3, -theta3]:
        gamma = math.atan2(z, x)
        beta = math.atan2(b * math.sin(theta3_sol), a + b * math.cos(theta3_sol))
        theta2 = gamma - beta
        x_check = a * math.cos(theta2) + b * math.cos(theta2 + theta3_sol)
        z_check = a * math.sin(theta2) + b * math.sin(theta2 + theta3_sol)
        if math.isclose(x_check, x, abs_tol=1e-05) and math.isclose(z_check, z, abs_tol=1e-05):
            solutions.append((theta2, theta3_sol))
    if not solutions:
        theta2, theta3 = (0.0, 0.0)
    else:
        theta2, theta3 = solutions[0]
    R_target = np.array([[math.cos(yaw) * math.cos(pitch), math.cos(yaw) * math.sin(pitch) * math.sin(roll) - math.sin(yaw) * math.cos(roll), math.cos(yaw) * math.sin(pitch) * math.cos(roll) + math.sin(yaw) * math.sin(roll)], [math.sin(yaw) * math.cos(pitch), math.sin(yaw) * math.sin(pitch) * math.sin(roll) + math.cos(yaw) * math.cos(roll), math.sin(yaw) * math.sin(pitch) * math.cos(roll) - math.cos(yaw) * math.sin(roll)], [-math.sin(pitch), math.cos(pitch) * math.sin(roll), math.cos(pitch) * math.cos(roll)]])
    R_j1 = np.array([[cos_theta1, -sin_theta1, 0], [sin_theta1, cos_theta1, 0], [0, 0, 1]])
    R_j2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
    R_j3 = np.array([[math.cos(theta3), 0, math.sin(theta3)], [0, 1, 0], [-math.sin(theta3), 0, math.cos(theta3)]])
    R_arm = R_j1 @ R_j2 @ R_j3
    R_remaining = R_arm.T @ R_target
    theta5 = math.atan2(R_remaining[1, 0], R_remaining[0, 0])
    theta4 = -(theta2 + theta3)
    return (theta1, theta2, theta3, theta4, theta5)