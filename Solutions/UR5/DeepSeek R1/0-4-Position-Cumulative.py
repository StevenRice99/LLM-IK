import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    D = 0.13585 - 0.1197
    L1 = 0.425
    a = 0.39225
    tcp_x_offset = 0.09465
    tcp_y_offset = 0.093
    tcp_z_offset = 0.09465
    solutions = []
    for theta4 in np.linspace(-np.pi, np.pi, 200):
        x_joint4 = x_target - tcp_x_offset * np.sin(theta4)
        y_joint4 = y_target - tcp_y_offset
        z_joint4 = z_target - tcp_z_offset * np.cos(theta4)
        try:
            numerator = x_joint4 ** 2 + y_joint4 ** 2 + z_joint4 ** 2 - D ** 2 - L1 ** 2 - a ** 2
            denominator = 2 * L1 * a
            if abs(denominator) < 1e-06:
                continue
            cos_theta3 = numerator / denominator
            cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
            theta3_1 = np.arccos(cos_theta3)
            theta3_2 = -theta3_1
            for theta3 in [theta3_1, theta3_2]:
                C1 = L1 + a * np.cos(theta3)
                C2 = a * np.sin(theta3)
                A_squared = x_joint4 ** 2 + y_joint4 ** 2 - D ** 2
                if A_squared < 0:
                    continue
                for A_sign in [1, -1]:
                    A = A_sign * np.sqrt(A_squared)
                    denom = C1 ** 2 + C2 ** 2
                    if denom == 0:
                        continue
                    sin_theta2 = (C1 * A - C2 * z_joint4) / denom
                    cos_theta2 = (C2 * A + C1 * z_joint4) / denom
                    if abs(sin_theta2) > 1 or abs(cos_theta2) > 1:
                        continue
                    theta2 = np.arctan2(sin_theta2, cos_theta2)
                    denom_theta1 = A ** 2 + D ** 2
                    if denom_theta1 == 0:
                        continue
                    cos_theta1 = (A * x_joint4 + D * y_joint4) / denom_theta1
                    sin_theta1 = (-D * x_joint4 + A * y_joint4) / denom_theta1
                    theta1 = np.arctan2(sin_theta1, cos_theta1)
                    solutions.append((theta1, theta2, theta3, theta4))
        except ValueError:
            continue
    if not solutions:
        raise ValueError('No valid solutions found')
    theta1, theta2, theta3, theta4 = min(solutions, key=lambda s: sum((abs(j) for j in s)))
    theta5 = 0.0
    return (theta1, theta2, theta3, theta4, theta5)