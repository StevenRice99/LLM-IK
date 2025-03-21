import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    wrist_offset = np.array([0, 0.1753, 0.09465])
    wx, wy, wz = np.array(p) - wrist_offset
    theta1 = np.arctan2(wy, wx)
    theta1_alt = theta1 + np.pi
    for theta1 in [theta1, theta1_alt]:
        x2 = 0.13585 * np.sin(theta1)
        y2 = 0.13585 * np.cos(theta1)
        z2 = 0.0
        dx = wx - x2
        dy = wy - y2
        dz = wz - z2
        dist = np.hypot(np.hypot(dx, dy), dz)
        a = np.sqrt(0.1197 ** 2 + 0.425 ** 2)
        b = 0.39225
        if not abs(a - b) <= dist <= a + b:
            continue
        cos_gamma = (a ** 2 + b ** 2 - dist ** 2) / (2 * a * b)
        gamma = np.arccos(np.clip(cos_gamma, -1, 1))
        for theta3 in [np.pi - gamma, gamma - np.pi]:
            sin_alpha = a * np.sin(gamma) / dist
            alpha = np.arcsin(np.clip(sin_alpha, -1, 1))
            theta2 = np.arctan2(dz, np.hypot(dx, dy)) - alpha * np.sign(theta3)
            if not (-np.pi <= theta2 <= np.pi and -np.pi <= theta3 <= np.pi):
                continue
            theta4 = -theta1 - theta2 - theta3
            theta5 = np.pi / 2
            theta6 = 0.0
            return (theta1, theta2, theta3, theta4, theta5, theta6)
    theta1 = np.arctan2(wy, wx)
    x2 = 0.13585 * np.sin(theta1)
    y2 = 0.13585 * np.cos(theta1)
    dx = wx - x2
    dy = wy - y2
    dz = wz
    dist = np.hypot(np.hypot(dx, dy), dz)
    a = np.sqrt(0.1197 ** 2 + 0.425 ** 2)
    b = 0.39225
    cos_gamma = (a ** 2 + b ** 2 - dist ** 2) / (2 * a * b)
    gamma = np.arccos(np.clip(cos_gamma, -1, 1))
    theta3 = np.pi - gamma
    sin_alpha = a * np.sin(gamma) / dist
    alpha = np.arcsin(np.clip(sin_alpha, -1, 1))
    theta2 = np.arctan2(dz, np.hypot(dx, dy)) - alpha
    theta4 = -theta1 - theta2 - theta3
    theta5 = np.pi / 2
    theta6 = 0.0
    return (theta1, theta2, theta3, theta4, theta5, theta6)