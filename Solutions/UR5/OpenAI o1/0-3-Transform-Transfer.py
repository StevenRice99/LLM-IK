def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Closed-form solution for the inverse kinematics of a 4-DOF arm with joints:
      - Joint1 rotates about Z
      - Joint2 rotates about Y
      - Joint3 rotates about Y
      - Joint4 rotates about Y
    The tool (TCP) position is at [0, 0, 0.093] from the last joint.

    We assume the only feasible orientation for the end-effector is given by yaw=q1
    and pitch=q2+q3+q4 (roll is not achievable). The function ignores roll (r[0]) 
    from the provided orientation and uses only (pitch, yaw).

    :param p: Target position [x, y, z].
    :param r: Target orientation [roll, pitch, yaw], where roll is ignored. 
    :return: (q1, q2, q3, q4) in radians.
    """
    import numpy as np
    x, y, z = p
    roll, pitch, yaw = r
    q1 = yaw
    c1 = np.cos(-q1)
    s1 = np.sin(-q1)
    X = x * c1 - y * s1
    Y = x * s1 + y * c1
    Z = z
    numerator = X ** 2 + Y ** 2 + Z ** 2 - 0.346395
    denominator = 0.3341625
    import math
    cos_q3 = numerator / denominator
    cos_q3 = max(-1.0, min(1.0, cos_q3))
    q3_temp = math.acos(cos_q3)
    q3 = q3_temp
    A = 0.425 + 0.39225 * math.cos(q3)
    B = 0.39225 * math.sin(q3)
    offset_val = 0.10915
    S_squared = X ** 2 + Y ** 2 - offset_val ** 2
    S = math.sqrt(S_squared) if S_squared >= 0 else 0.0
    C = Z
    numerator_q2 = S * A - C * B
    denominator_q2 = S * B + C * A
    q2 = math.atan2(numerator_q2, denominator_q2)
    phi = math.atan2(offset_val, S) if abs(S) > 1e-09 else math.pi / 2.0
    q4 = pitch - (q2 + q3)

    def wrap_to_pi(angle):
        return math.atan2(math.sin(angle), math.cos(angle))
    q1 = wrap_to_pi(q1)
    q2 = wrap_to_pi(q2)
    q3 = wrap_to_pi(q3)
    q4 = wrap_to_pi(q4)
    return (q1, q2, q3, q4)