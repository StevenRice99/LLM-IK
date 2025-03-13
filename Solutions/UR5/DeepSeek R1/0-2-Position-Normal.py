import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    B = 0.01615
    L1 = 0.425
    L2 = 0.39225
    D = x ** 2 + y ** 2 + z ** 2
    cos_theta3 = (D - 0.334746) / 0.33365625
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    theta3_solutions = [theta3, -theta3]
    for theta3 in theta3_solutions:
        A_squared = x ** 2 + y ** 2 - B ** 2
        if A_squared < 0:
            continue
        for A_sign in [1, -1]:
            A = A_sign * np.sqrt(A_squared)
            C = L1 + L2 * np.cos(theta3)
            S = L2 * np.sin(theta3)
            denom = C ** 2 + S ** 2
            if denom == 0:
                continue
            sin_theta2 = (C * A - S * z) / denom
            cos_theta2 = (S * A + C * z) / denom
            theta2 = np.arctan2(sin_theta2, cos_theta2)
            denom_theta1 = A ** 2 + B ** 2
            if denom_theta1 == 0:
                continue
            cos_theta1 = (A * x + B * y) / denom_theta1
            sin_theta1 = (-B * x + A * y) / denom_theta1
            theta1 = np.arctan2(sin_theta1, cos_theta1)
            return (theta1, theta2, theta3)
    return (0.0, 0.0, 0.0)