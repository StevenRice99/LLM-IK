import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    y_adj = y_target - 0.093
    numerator = x_target ** 2 + y_adj ** 2 + z_target ** 2 - (0.425 ** 2 + 0.39225 ** 2)
    denominator = 2 * 0.425 * 0.39225
    cos_q3 = numerator / denominator
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = np.arccos(cos_q3)
    A = 0.425 + 0.39225 * np.cos(q3)
    B = 0.39225 * np.sin(q3)
    S_squared = x_target ** 2 + y_adj ** 2 - 0.10915 ** 2
    S = np.sqrt(S_squared) if S_squared >= 0 else 0.0
    C = z_target
    numerator_q2 = S * A - C * B
    denominator_q2 = S * B + C * A
    q2 = np.arctan2(numerator_q2, denominator_q2)
    phi = np.arctan2(0.10915, S) if S != 0 else np.pi / 2
    q1 = np.arctan2(y_adj, x_target) - phi
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    c1, s1 = (np.cos(q1), np.sin(q1))
    c2, s2 = (np.cos(q2), np.sin(q2))
    c3, s3 = (np.cos(q3), np.sin(q3))
    x_after_q2 = s2 * (0.425 + 0.39225 * c3) + 0.39225 * s3 * c2
    z_after_q2 = c2 * (0.425 + 0.39225 * c3) - 0.39225 * s3 * s2
    x_link4 = x_after_q2 * c1 - 0.01615 * s1
    z_link4 = z_after_q2
    delta_x = x_target - x_link4
    delta_z = z_target - z_link4 - 0.09465
    q4 = np.arctan2(delta_x, delta_z)
    q5 = 0.0
    return (q1, q2, q3, q4, q5)