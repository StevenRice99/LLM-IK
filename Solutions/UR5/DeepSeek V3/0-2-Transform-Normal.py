import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    phi, theta, psi = r
    theta1 = np.arctan2(y, x)
    x_prime = np.sqrt(x ** 2 + y ** 2)
    z_prime = z - 0.13585
    d = np.sqrt(x_prime ** 2 + z_prime ** 2)
    L2 = 0.425
    L3 = 0.39225
    alpha = np.arctan2(z_prime, x_prime)
    beta = np.arccos(np.clip((L2 ** 2 + d ** 2 - L3 ** 2) / (2 * L2 * d), -1, 1))
    theta2 = alpha - beta
    theta3 = np.arccos(np.clip((L2 ** 2 + L3 ** 2 - d ** 2) / (2 * L2 * L3), -1, 1))
    theta2_alt = alpha + beta
    theta3_alt = -theta3
    return (theta1, theta2, theta3)