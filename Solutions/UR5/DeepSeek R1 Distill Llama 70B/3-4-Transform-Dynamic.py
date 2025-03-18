import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    L1 = 0.093
    L2 = 0.09465
    px, py, pz = p
    cos_theta2 = (px ** 2 + py ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    D = (px ** 2 + py ** 2 + L2 ** 2 - L1 ** 2) / (2 * L2)
    A = px
    B = py
    delta = math.atan2(B, A)
    magnitude = math.sqrt(A ** 2 + B ** 2)
    if magnitude == 0:
        phi = 0.0
    else:
        cos_phi_delta = D / magnitude
        cos_phi_delta = max(min(cos_phi_delta, 1.0), -1.0)
        phi_delta = math.acos(cos_phi_delta)
        phi = delta + phi_delta
    theta1 = phi - theta2
    return (theta1, theta2)