import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = math.atan2(y, x)
    d1 = 0.13585
    a2 = 0.425
    a3 = 0.39225
    d4 = 0.093
    d6 = 0.09465
    x_wrist = x - d6 * math.cos(pitch) * math.cos(yaw)
    y_wrist = y - d6 * math.cos(pitch) * math.sin(yaw)
    z_wrist = z - d6 * math.sin(pitch)
    r = math.sqrt(x_wrist ** 2 + y_wrist ** 2)
    D = (r ** 2 + (z_wrist - d1) ** 2 - a2 ** 2 - a3 ** 2) / (2 * a2 * a3)
    D = max(-1, min(1, D))
    theta3 = math.atan2(math.sqrt(1 - D ** 2), D)
    theta2 = math.atan2(z_wrist - d1, r) - math.atan2(a3 * math.sin(theta3), a2 + a3 * math.cos(theta3))
    R0_3 = np.array([[math.cos(theta1) * math.cos(theta2 + theta3), -math.cos(theta1) * math.sin(theta2 + theta3), math.sin(theta1)], [math.sin(theta1) * math.cos(theta2 + theta3), -math.sin(theta1) * math.sin(theta2 + theta3), -math.cos(theta1)], [math.sin(theta2 + theta3), math.cos(theta2 + theta3), 0]])
    R0_6 = np.array([[math.cos(yaw) * math.cos(pitch), -math.sin(yaw) * math.cos(roll) + math.cos(yaw) * math.sin(pitch) * math.sin(roll), math.sin(yaw) * math.sin(roll) + math.cos(yaw) * math.sin(pitch) * math.cos(roll)], [math.sin(yaw) * math.cos(pitch), math.cos(yaw) * math.cos(roll) + math.sin(yaw) * math.sin(pitch) * math.sin(roll), -math.cos(yaw) * math.sin(roll) + math.sin(yaw) * math.sin(pitch) * math.cos(roll)], [-math.sin(pitch), math.cos(pitch) * math.sin(roll), math.cos(pitch) * math.cos(roll)]])
    R3_6 = np.dot(np.linalg.inv(R0_3), R0_6)
    theta5 = math.atan2(math.sqrt(R3_6[0, 2] ** 2 + R3_6[1, 2] ** 2), R3_6[2, 2])
    theta4 = math.atan2(R3_6[1, 2], R3_6[0, 2])
    theta6 = math.atan2(R3_6[2, 1], -R3_6[2, 0])
    return (theta1, theta2, theta3, theta4, theta5, theta6)