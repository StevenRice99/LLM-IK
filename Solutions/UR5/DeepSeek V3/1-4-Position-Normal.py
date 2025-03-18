import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    d1 = 0.425
    d2 = 0.39225
    d3 = 0.093
    d4 = 0.09465
    L = d1 + d2 + d3 + d4
    distance_to_target = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    if distance_to_target > L:
        raise ValueError('Target position is unreachable.')
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    d = np.sqrt(r ** 2 + (z - d1) ** 2)
    cos_theta3 = (d2 ** 2 + d3 ** 2 - d ** 2) / (2 * d2 * d3)
    theta3 = np.arccos(np.clip(cos_theta3, -1, 1))
    cos_theta2 = (d2 ** 2 + d ** 2 - d3 ** 2) / (2 * d2 * d)
    theta2 = np.arctan2(z - d1, r) - np.arccos(np.clip(cos_theta2, -1, 1))
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)