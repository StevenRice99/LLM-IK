import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x, y, z = p
    theta1 = np.arctan2(y, x)
    for _ in range(3):
        x_r2 = 0.13585 * np.sin(theta1)
        y_r2 = 0.13585 * np.cos(theta1)
        dx = x - x_r2
        dy = y - y_r2
        theta1 = np.arctan2(dy, dx)
    x_r2 = 0.13585 * np.sin(theta1)
    y_r2 = 0.13585 * np.cos(theta1)
    z_r2 = 0.0
    dx = x - x_r2
    dy = y - y_r2
    dz = z - z_r2
    r = np.hypot(dx, dy)
    h = dz - 0.1197
    L1 = 0.425
    L2 = 0.39225
    L3 = np.hypot(0.093, 0.09465)
    eff_length = L2 + L3
    D = np.hypot(r, h)
    cos_theta3 = (D ** 2 - L1 ** 2 - eff_length ** 2) / (2 * L1 * eff_length)
    theta3 = -np.arccos(np.clip(cos_theta3, -1.0, 1.0))
    alpha = np.arctan2(h, r)
    beta = np.arcsin(eff_length * np.sin(-theta3) / D)
    theta2 = alpha - beta
    theta4 = -theta2 - theta3 - np.arctan2(0.093, 0.09465)
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
    theta3 = (theta3 + np.pi) % (2 * np.pi) - np.pi
    theta4 = (theta4 + np.pi) % (2 * np.pi) - np.pi
    return (theta1, theta2, theta3, theta4, 0.0)