import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    px, py, pz = p
    rx, ry, rz = r
    l1 = 0.09465
    l2 = 0.0823
    max_z = l1 + l2
    min_z = abs(l1 - l2)
    pz_clamped = min(max(pz, -max_z), max_z)
    epsilon = 1e-12
    theta2 = math.asin(max(min(pz_clamped / l2, 1.0 - epsilon), -1.0 + epsilon))
    A1 = l1 * math.cos(theta2)
    B1 = -l1 * math.sin(theta2) + l2
    A2 = l1 * math.cos(theta2) + l2
    B2 = l1 * math.sin(theta2)
    det = A1 * B2 - A2 * B1
    if abs(det) < epsilon:
        det = epsilon
    sin_phi = (B2 * px - A2 * py) / det
    cos_phi = (A1 * py - B1 * px) / det
    phi = math.atan2(sin_phi, cos_phi)
    theta1 = phi - theta2
    return (theta1, theta2)