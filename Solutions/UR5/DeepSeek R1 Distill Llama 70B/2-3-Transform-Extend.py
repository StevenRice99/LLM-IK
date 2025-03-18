import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of the values to set the joints to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    rx, ry, rz = r
    L1 = 0.39225
    L2 = 0.093
    rz = -rz
    cos_q1 = (x - L2 * math.cos(rz)) / L1
    sin_q1 = (z - L2 * math.sin(rz)) / L1
    q1 = math.atan2(sin_q1, cos_q1)
    q2 = rz - q1
    return (q1, q2)