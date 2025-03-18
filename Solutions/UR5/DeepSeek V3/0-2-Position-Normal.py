import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    x, y, z = p
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    L2 = 0.13585
    L3 = 0.425
    L_TCP = 0.39225
    z_eff = z - L_TCP
    D = (r ** 2 + z_eff ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    D = np.clip(D, -1.0, 1.0)
    theta3_1 = np.arccos(D)
    theta3_2 = -np.arccos(D)

    def solve_theta2(theta3):
        return np.arctan2(z_eff, r) - np.arctan2(L3 * np.sin(theta3), L2 + L3 * np.cos(theta3))
    theta2_1 = solve_theta2(theta3_1)
    theta2_2 = solve_theta2(theta3_2)
    return (theta1, theta2_1, theta3_1)