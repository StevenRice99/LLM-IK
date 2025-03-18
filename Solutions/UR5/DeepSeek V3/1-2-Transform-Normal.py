import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    x, y, z = p
    r_x, r_y, r_z = r
    theta1 = np.arctan2(x, z)
    d = np.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (d ** 2 + 0.425 ** 2 - 0.39225 ** 2) / (2 * d * 0.425)
    theta2 = np.arccos(cos_theta2)
    theta2 = r_y - theta1
    return (theta1, theta2)