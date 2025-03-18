import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    x, y, z = p
    theta1 = np.arctan2(y, x)
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    x_prime = np.sqrt(x ** 2 + y ** 2)
    z_prime = z - L1
    D = (x_prime ** 2 + z_prime ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    if D < -1 or D > 1:
        raise ValueError('Target position is unreachable.')
    theta3 = np.arctan2(np.sqrt(1 - D ** 2), D)
    alpha = np.arctan2(z_prime, x_prime)
    beta = np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
    theta2 = alpha - beta
    theta4 = 0.0
    theta5 = 0.0
    theta6 = 0.0
    return (theta1, theta2, theta3, theta4, theta5, theta6)