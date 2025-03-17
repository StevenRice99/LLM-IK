import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x, y, z = p
    D = 0.13585 - 0.1197
    L1 = 0.425
    a = 0.39225 + 0.09465
    numerator = x ** 2 + y ** 2 + z ** 2 - D ** 2 - L1 ** 2 - a ** 2
    denominator = 2 * L1 * a
    if abs(denominator) < 1e-06:
        raise ValueError('Denominator is zero')
    cos_theta3 = numerator / denominator
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3_1 = np.arccos(cos_theta3)
    theta3_2 = -theta3_1
    solutions = []
    for theta3 in [theta3_1, theta3_2]:
        C1 = L1 + a * np.cos(theta3)
        C2 = a * np.sin(theta3)
        A_squared = x ** 2 + y ** 2 - D ** 2
        if A_squared < 0:
            continue
        for A_sign in [1, -1]:
            A = A_sign * np.sqrt(A_squared)
            denom = C1 ** 2 + C2 ** 2
            if denom == 0:
                continue
            sin_theta2 = (C1 * A - C2 * z) / denom
            cos_theta2 = (C2 * A + C1 * z) / denom
            if abs(sin_theta2) > 1 or abs(cos_theta2) > 1:
                continue
            theta2 = np.arctan2(sin_theta2, cos_theta2)
            denom_theta1 = A ** 2 + D ** 2
            if denom_theta1 == 0:
                continue
            cos_theta1 = (A * x + D * y) / denom_theta1
            sin_theta1 = (-D * x + A * y) / denom_theta1
            theta1 = np.arctan2(sin_theta1, cos_theta1)
            theta4 = -(theta2 + theta3)
            solutions.append((theta1, theta2, theta3, theta4))
    if not solutions:
        raise ValueError('No valid solutions found')
    theta1, theta2, theta3, theta4 = min(solutions, key=lambda s: abs(s[2]))
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)