import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    x_d, y_d, z_d = p
    L1 = 0.425
    L2 = 0.39225
    C = (L1 ** 2 + L2 ** 2 - x_d ** 2 - z_d ** 2) / (2 * L1)
    A = x_d
    B = z_d
    discriminant = A ** 2 + B ** 2 - C ** 2
    if discriminant < 0:
        raise ValueError('No real solution exists for the given target position.')
    sqrt_discriminant = np.sqrt(discriminant)
    cos_theta1_1 = (B * C + A * sqrt_discriminant) / (A ** 2 + B ** 2)
    cos_theta1_2 = (B * C - A * sqrt_discriminant) / (A ** 2 + B ** 2)
    valid_cos_theta1 = []
    for cos_val in [cos_theta1_1, cos_theta1_2]:
        if -1 <= cos_val <= 1:
            valid_cos_theta1.append(cos_val)
    if not valid_cos_theta1:
        raise ValueError('No valid solution for cos(theta1) within [-1, 1].')
    cos_theta1 = valid_cos_theta1[0]
    theta1 = np.arccos(cos_theta1)
    sin_theta1 = np.sqrt(1 - cos_theta1 ** 2)
    sin_alpha = (x_d - L1 * sin_theta1) / L2
    cos_alpha = (L1 * cos_theta1 - z_d) / L2
    alpha = np.arctan2(sin_alpha, cos_alpha)
    theta2 = alpha - theta1
    return (theta1, theta2)