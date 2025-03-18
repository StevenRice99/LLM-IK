import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x, y, z = p
    theta1 = np.arctan2(z, x)
    r = np.sqrt(x ** 2 + z ** 2)
    h = y - -0.1197
    a = 0.425
    b = 0.39225
    c = np.sqrt(r ** 2 + h ** 2)
    alpha = np.arccos(np.clip((a ** 2 + c ** 2 - b ** 2) / (2 * a * c), -1, 1))
    beta = np.arccos(np.clip((a ** 2 + b ** 2 - c ** 2) / (2 * a * b), -1, 1))
    theta2 = np.arctan2(h, r) - alpha
    theta3 = np.pi - beta
    theta4 = 0.0
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)