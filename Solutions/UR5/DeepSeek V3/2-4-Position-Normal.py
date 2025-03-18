import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    theta3 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    z_prime = z - L1
    cos_theta2 = (r ** 2 + z_prime ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    theta2 = np.arccos(cos_theta2)
    alpha = np.arctan2(z_prime, r)
    beta = np.arctan2(L3 * np.sin(theta2), L2 + L3 * np.cos(theta2))
    theta1 = alpha - beta
    return (theta1, theta2, theta3)