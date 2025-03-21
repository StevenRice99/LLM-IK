import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    x_tcp, y_tcp, z_tcp = p
    d1 = 0.13585
    a2 = 0.425
    a3 = 0.39225
    d5 = 0.093
    d6 = 0.09465
    tcp_offset = 0.0823
    displacement_local = np.array([0, d5 + tcp_offset, d6])
    wrist_center_guess = (x_tcp, y_tcp, z_tcp)
    theta1, theta2, theta3 = solve_position(wrist_center_guess, d1, a2, a3)
    R03 = compute_orientation_matrix(theta1, theta2, theta3)
    displacement_world = R03 @ displacement_local
    wrist_center = (x_tcp - displacement_world[0], y_tcp - displacement_world[1], z_tcp - displacement_world[2])
    theta1, theta2, theta3 = solve_position(wrist_center, d1, a2, a3)
    R03 = compute_orientation_matrix(theta1, theta2, theta3)
    R_tcp = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R36 = R03.T @ R_tcp
    theta4, theta5, theta6 = yzy_euler_angles(R36)
    return (theta1, theta2, theta3, theta4, theta5, theta6)

def solve_position(p, d1, a2, a3):
    x, y, z = p
    theta1 = math.atan2(y, x)
    x_proj = math.hypot(x, y)
    z_adj = z - d1
    D = (x_proj ** 2 + z_adj ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
    D = np.clip(D, -1.0, 1.0)
    theta3 = math.acos(D)
    alpha = math.atan2(z_adj, x_proj)
    beta = math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)

def compute_orientation_matrix(theta1, theta2, theta3):
    Rz = np.array([[math.cos(theta1), -math.sin(theta1), 0], [math.sin(theta1), math.cos(theta1), 0], [0, 0, 1]])
    Ry2 = np.array([[math.cos(theta2), 0, math.sin(theta2)], [0, 1, 0], [-math.sin(theta2), 0, math.cos(theta2)]])
    Ry3 = np.array([[math.cos(theta3), 0, math.sin(theta3)], [0, 1, 0], [-math.sin(theta3), 0, math.cos(theta3)]])
    return Rz @ Ry2 @ Ry3

def yzy_euler_angles(R):
    epsilon = 1e-06
    if abs(R[1, 1] - 1.0) < epsilon:
        theta5 = 0.0
        theta4 = math.atan2(R[2, 0], R[0, 0])
        theta6 = 0.0
    elif abs(R[1, 1] + 1.0) < epsilon:
        theta5 = math.pi
        theta4 = math.atan2(-R[2, 0], R[0, 0])
        theta6 = 0.0
    else:
        theta5 = math.acos(R[1, 1])
        theta4 = math.atan2(R[2, 1] / math.sin(theta5), -R[0, 1] / math.sin(theta5))
        theta6 = math.atan2(R[1, 2] / math.sin(theta5), R[1, 0] / math.sin(theta5))
    return (theta4, theta5, theta6)