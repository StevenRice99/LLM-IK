import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    x, y, z = p
    numerator = x ** 2 + y ** 2 + z ** 2 - 0.346395
    denominator = 0.3341625
    cos_q3 = numerator / denominator
    cos_q3 = np.clip(cos_q3, -1.0, 1.0)
    q3 = np.arccos(cos_q3)
    A = 0.425 + 0.39225 * np.cos(q3)
    B = 0.39225 * np.sin(q3)
    S_squared = x ** 2 + y ** 2 - 0.10915 ** 2
    S = np.sqrt(S_squared) if S_squared >= 0 else 0.0
    C = z
    numerator_q2 = S * A - C * B
    denominator_q2 = S * B + C * A
    q2 = np.arctan2(numerator_q2, denominator_q2)
    phi = np.arctan2(0.10915, S) if S != 0 else np.pi / 2
    q1 = np.arctan2(y, x) - phi
    q1 = (q1 + np.pi) % (2 * np.pi) - np.pi
    q4 = 0.0
    return (q1, q2, q3, q4)