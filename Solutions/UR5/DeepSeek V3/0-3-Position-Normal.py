import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    z_target = z
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    L4 = 0.093
    x_w = r - L4 * np.cos(theta1)
    y_w = z_target - L4 * np.sin(theta1)
    d = np.sqrt(x_w ** 2 + y_w ** 2)
    cos_theta3 = (d ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    theta3 = np.arccos(np.clip(cos_theta3, -1, 1))
    alpha = np.arctan2(y_w, x_w)
    beta = np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
    theta2 = alpha - beta
    theta4 = -(theta2 + theta3)
    return (theta1, theta2, theta3, theta4)