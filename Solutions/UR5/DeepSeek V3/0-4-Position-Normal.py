import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x, y, z = p
    L2 = 0.13585
    L3 = 0.425
    L4 = 0.39225
    L5 = 0.093
    L6 = 0.09465
    theta1 = np.arctan2(y, x)
    x_w = x - L6 * np.cos(theta1)
    y_w = y - L6 * np.sin(theta1)
    z_w = z
    theta2 = np.arctan2(z_w, np.sqrt(x_w ** 2 + y_w ** 2))
    theta3 = np.arctan2(z_w - L2 * np.sin(theta2), np.sqrt(x_w ** 2 + y_w ** 2) - L2 * np.cos(theta2))
    theta4 = np.arctan2(z_w - L2 * np.sin(theta2) - L3 * np.sin(theta3), np.sqrt(x_w ** 2 + y_w ** 2) - L2 * np.cos(theta2) - L3 * np.cos(theta3))
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)