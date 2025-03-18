import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    z_total = 0.39225
    z_tcp_offset = 0.09465
    z_desired = pz - z_tcp_offset
    z_desired = max(min(z_desired, z_total), -z_total)
    cos_q1 = z_desired / z_total
    cos_q1 = max(min(cos_q1, 1.0), -1.0)
    q1 = math.acos(cos_q1)
    A = -0.093 * math.sin(q1)
    B = -0.093
    denominator = A ** 2 + B ** 2
    if denominator == 0:
        q3 = 0.0
    else:
        cos_q3 = (A * px + B * py) / denominator
        sin_q3 = (A * py - B * px) / denominator
        cos_q3 = max(min(cos_q3, 1.0), -1.0)
        sin_q3 = max(min(sin_q3, 1.0), -1.0)
        q3 = math.atan2(sin_q3, cos_q3)
    roll, pitch, yaw = r
    q2 = 0.0
    return (q1, q2, q3)