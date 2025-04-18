import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Analytic inverse kinematics for the 5‑DOF arm.
    :param p: target TCP position [x, y, z]
    :param r: target TCP orientation [roll, pitch, yaw]
    :return: (theta1, theta2, theta3, theta4, theta5)
    """
    d1 = 0.425
    d2 = 0.39225
    d4 = 0.09465
    d_tcp = 0.0823
    y_off_low = -0.1197
    y_off_high = +0.093
    psi = 1.570796325
    x, y, z = p
    roll, pitch, yaw = r

    def rot_x(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def rot_z(a):
        ca, sa = (math.cos(a), math.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    R_target = rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)
    phi = math.atan2(R_target[0, 2], R_target[2, 2])
    cos4 = (y - (y_off_low + y_off_high)) / d_tcp
    cos4 = max(min(cos4, 1.0), -1.0)
    theta4 = math.acos(cos4)
    sin4 = math.sin(theta4)
    X = x - d4 * math.sin(phi) + d_tcp * sin4 * math.cos(phi)
    Z = z - d4 * math.cos(phi) - d_tcp * sin4 * math.sin(phi)
    D = (X * X + Z * Z - d1 * d1 - d2 * d2) / (2 * d1 * d2)
    D = max(min(D, 1.0), -1.0)
    theta2 = math.acos(D)
    theta1 = math.atan2(X, Z) - math.atan2(d2 * math.sin(theta2), d1 + d2 * math.cos(theta2))
    theta3 = phi - theta1 - theta2
    R_pre = rot_y(phi)
    R_invpre = R_pre.T
    R_post = R_target @ rot_z(-psi)
    R45 = R_pre.T @ R_post
    R5 = rot_z(-theta4) @ R45
    sin5 = R5[0, 2]
    cos5 = R5[0, 0]
    theta5 = math.atan2(sin5, cos5)
    return (theta1, theta2, theta3, theta4, theta5)