import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x_target, y_target, z_target = p
    L1 = 0.425
    L2 = 0.39225
    L3 = np.hypot(0.093, 0.09465)
    gamma = np.arctan2(0.093, 0.09465)
    theta = np.arctan2(x_target, z_target)
    D = np.hypot(x_target, z_target)
    D_eff = D - L3 * np.cos(gamma)
    a = L1
    b = np.hypot(L2, L3 * np.sin(gamma))
    cos_q2 = (D_eff ** 2 - a ** 2 - b ** 2) / (2 * a * b)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2 = np.arccos(cos_q2)
    q2_1 = q2
    q2_2 = -q2
    alpha_1 = np.arctan2(b * np.sin(q2_1), a + b * np.cos(q2_1))
    q1_1 = theta - alpha_1
    alpha_2 = np.arctan2(b * np.sin(q2_2), a + b * np.cos(q2_2))
    q1_2 = theta - alpha_2

    def compute_position(q1, q2):
        x_r3 = L1 * np.sin(q1) + L2 * np.sin(q1 + q2)
        z_r3 = L1 * np.cos(q1) + L2 * np.cos(q1 + q2)
        dx = x_target - x_r3
        dz = z_target - z_r3
        beta = np.arctan2(dx, dz) - (q1 + q2)
        return beta
    beta_1 = compute_position(q1_1, q2_1)
    beta_2 = compute_position(q1_2, q2_2)
    if abs(beta_1) <= abs(beta_2):
        q1, q2, q3 = (q1_1, q2_1, beta_1)
    else:
        q1, q2, q3 = (q1_2, q2_2, beta_2)
    q3 -= gamma
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    q2 = (q2 + np.pi) % (2 * np.pi) - np.pi
    q3 = (q3 + np.pi) % (2 * np.pi) - np.pi
    return (q1, q2, q3, 0.0)