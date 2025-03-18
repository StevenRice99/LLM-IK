import numpy as np
from math import atan2, sqrt, acos, pi

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = atan2(y, x)
    x_prime = sqrt(x ** 2 + y ** 2)
    y_prime = z - 0.13585
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    d = sqrt(x_prime ** 2 + y_prime ** 2)
    cos_theta3 = (L1 ** 2 + L2 ** 2 - d ** 2) / (2 * L1 * L2)
    cos_theta3 = max(min(cos_theta3, 1), -1)
    theta3 = acos(cos_theta3)
    alpha = atan2(y_prime, x_prime)
    cos_beta = (L1 ** 2 + d ** 2 - L2 ** 2) / (2 * L1 * d)
    cos_beta = max(min(cos_beta, 1), -1)
    beta = acos(cos_beta)
    theta2 = alpha - beta
    theta4 = -theta2 - theta3
    theta5 = yaw
    theta6 = pitch
    return (theta1, theta2, theta3, theta4, theta5, theta6)