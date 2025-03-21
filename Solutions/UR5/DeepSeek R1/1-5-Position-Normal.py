import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    x_target, y_target, z_target = p
    y_tcp_contribution = 0.0823
    y_revolute4 = 0.093
    cos_q4 = (y_target - y_revolute4) / y_tcp_contribution
    cos_q4 = np.clip(cos_q4, -1.0, 1.0)
    q4_pos = np.arccos(cos_q4)
    q4_neg = -q4_pos
    x_pos_effect_pos = y_tcp_contribution * np.sin(q4_pos)
    x_pos_effect_neg = y_tcp_contribution * np.sin(q4_neg)
    if abs(x_target - x_pos_effect_pos) < abs(x_target - x_pos_effect_neg):
        q4 = q4_pos
    else:
        q4 = q4_neg
    x_adj = x_target - y_tcp_contribution * np.sin(q4)
    z_adj = z_target - 0.09465
    a1 = 0.425
    a2 = 0.39225
    a3 = 0.093
    planar_target = np.sqrt(x_adj ** 2 + z_adj ** 2)
    cos_q2 = (x_adj ** 2 + z_adj ** 2 - a1 ** 2 - a2 ** 2) / (2 * a1 * a2)
    cos_q2 = np.clip(cos_q2, -1.0, 1.0)
    q2 = np.arccos(cos_q2)
    q2 = -q2
    gamma = np.arctan2(z_adj, x_adj)
    delta = np.arctan2(a2 * np.sin(q2), a1 + a2 * np.cos(q2))
    q1 = gamma - delta
    x_current = a1 * np.sin(q1) + a2 * np.sin(q1 + q2)
    z_current = a1 * np.cos(q1) + a2 * np.cos(q1 + q2)
    dx = x_adj - x_current
    dz = z_adj - z_current
    q3 = np.arctan2(dx, dz) - (q1 + q2)
    q3 = np.arctan2(np.sin(q3), np.cos(q3))
    q5 = 0.0
    return (q1, q2, q3, q4, q5)