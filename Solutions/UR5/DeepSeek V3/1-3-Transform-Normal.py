import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    rx, ry, rz = r
    theta1 = np.arctan2(x, z)
    L1 = 0.425
    L2 = 0.39225
    d = np.sqrt(x ** 2 + z ** 2)
    cos_theta3 = (d ** 2 - L1 ^ 2 - L2 ^ 2) / (2 * L1 * L2)
    theta3 = np.arccos(cos_theta3)
    alpha = np.arctan2(z, x)
    beta = np.arctan2(L2 * np.sin(theta3), L1 + L2 * np.cos(theta3))
    theta2 = alpha - beta
    theta_sum = theta1 + theta2 + theta3
    if theta_sum < -np.pi:
        theta2 += 2 * np.pi
    elif theta_sum > np.pi:
        theta2 -= 2 * np.pi
    return (theta1, theta2, theta3)