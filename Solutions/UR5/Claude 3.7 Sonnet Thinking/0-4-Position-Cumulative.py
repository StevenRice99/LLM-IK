import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.09465
    j1_to_j2_y = 0.13585
    j2_to_j3_y = -0.1197
    j4_to_j5_y = 0.093
    Y_OFFSET = j1_to_j2_y + j2_to_j3_y + j4_to_j5_y
    z_adj = z - L3
    numerator = x ** 2 + y ** 2 + z_adj ** 2 - L1 ** 2 - L2 ** 2
    denominator = 2 * L1 * L2
    cos_q3 = numerator / denominator
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = -np.arccos(cos_q3)
    A = L1 + L2 * np.cos(q3)
    B = L2 * np.sin(q3)
    S_squared = x ** 2 + y ** 2 - Y_OFFSET ** 2
    S = np.sqrt(S_squared) if S_squared >= 0 else 0.0
    C = z_adj
    numerator_q2 = S * A - C * B
    denominator_q2 = S * B + C * A
    q2 = np.arctan2(numerator_q2, denominator_q2)
    phi = np.arctan2(Y_OFFSET, S) if S != 0 else np.pi / 2
    q1 = np.arctan2(y, x) - phi
    q4 = -(q2 + q3)
    q5 = 0.0

    def normalize_angle(angle):
        return (angle + np.pi) % (2 * np.pi) - np.pi
    q1 = normalize_angle(q1)
    q2 = normalize_angle(q2)
    q3 = normalize_angle(q3)
    q4 = normalize_angle(q4)
    return (q1, q2, q3, q4, q5)