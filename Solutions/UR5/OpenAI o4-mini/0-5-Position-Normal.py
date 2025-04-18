import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" (pure position solve).
    We fix q5 = q6 = 0 and choose q4 = -(q2+q3) to decouple the wrist,
    then solve a 2-link planar chain (d3,d4) for q2,q3, and finally q1.
    :param p: The desired TCP position [x, y, z].
    :return: (q1, q2, q3, q4, q5, q6) in radians.
    """
    x, y, z = p
    d1 = 0.13585
    b = 0.1197
    d3 = 0.425
    d4 = 0.39225
    d5 = 0.093
    d6 = 0.09465
    d7 = 0.0823
    D = d1 + d5 + d7 - b
    R = math.hypot(x, y)
    C = math.sqrt(max(R * R - D * D, 0.0))
    z_p = z - d6
    num = z_p * z_p + C * C - d3 * d3 - d4 * d4
    den = 2.0 * d3 * d4
    cos_q3 = max(-1.0, min(1.0, num / den))
    sin_q3 = math.sqrt(max(0.0, 1.0 - cos_q3 * cos_q3))
    q3 = math.atan2(sin_q3, cos_q3)
    phi2 = math.atan2(C, z_p)
    phi_k = math.atan2(d4 * sin_q3, d3 + d4 * cos_q3)
    q2 = phi2 - phi_k
    q4 = -(q2 + q3)
    q5 = 0.0
    q6 = 0.0
    q1 = math.atan2(y, x) - math.atan2(D, C)
    return (q1, q2, q3, q4, q5, q6)