import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    L1 = 0.1197
    L2 = 0.425
    L3 = 0.39225
    theta1 = np.arctan2(z, x)
    r = np.sqrt(x ** 2 + z ** 2)
    y_prime = y - L1
    d = np.sqrt(r ** 2 + y_prime ** 2)
    cos_theta3 = (L2 ** 2 + L3 ** 2 - d ** 2) / (2 * L2 * L3)
    theta3 = np.arccos(np.clip(cos_theta3, -1, 1))
    alpha = np.arctan2(y_prime, r)
    beta = np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
    theta2 = alpha - beta
    return (theta1, theta2, theta3)