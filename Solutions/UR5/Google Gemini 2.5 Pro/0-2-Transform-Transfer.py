import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the joint values (q1, q2, q3) in radians.
    """
    Px, Py, Pz = p
    phi, theta, psi = r
    d1y = 0.13585
    d2y = -0.1197
    d2z = 0.425
    d3z = 0.39225
    cphi = math.cos(phi)
    sphi = math.sin(phi)
    cth = math.cos(theta)
    sth = math.sin(theta)
    cpsi = math.cos(psi)
    spsi = math.sin(psi)
    r12 = cpsi * sth * sphi - spsi * cphi
    r22 = spsi * sth * sphi + cpsi * cphi
    r31 = -sth
    r33 = cth * cphi
    q1 = math.atan2(-r12, r22)
    s1 = math.sin(q1)
    c1 = math.cos(q1)
    S_q23 = math.atan2(-r31, r33)
    s_S_q23 = math.sin(S_q23)
    c_S_q23 = math.cos(S_q23)
    if d2z == 0:
        raise ValueError('d2z cannot be zero for this solution method.')
    c2_val_numerator = Pz - c_S_q23 * d3z
    c2_val = c2_val_numerator / d2z
    s2_val_numerator = c1 * Px + s1 * Py - s_S_q23 * d3z
    s2_val = s2_val_numerator / d2z
    c2_val = max(min(c2_val, 1.0), -1.0)
    s2_val = max(min(s2_val, 1.0), -1.0)
    q2 = math.atan2(s2_val, c2_val)
    q3 = S_q23 - q2
    return (q1, q2, q3)