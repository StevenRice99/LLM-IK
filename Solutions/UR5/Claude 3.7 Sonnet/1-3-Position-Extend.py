import math
import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x_target = p[0]
    y_target = p[1]
    z_target = p[2]
    l1 = 0.425
    l2 = 0.39225
    y_offset = -0.1197
    tcp_offset = 0.093
    y_joint3 = y_target - tcp_offset
    x_joint3 = x_target
    z_joint3 = z_target
    r = math.sqrt(x_joint3 ** 2 + z_joint3 ** 2)
    h = y_joint3 - y_offset
    d = math.sqrt(r ** 2 + h ** 2)
    if d > l1 + l2:
        cos_theta2 = -1.0
    elif d < abs(l1 - l2):
        cos_theta2 = 1.0
    else:
        cos_theta2 = (l1 ** 2 + l2 ** 2 - d ** 2) / (2 * l1 * l2)
        cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    phi_xz = math.atan2(x_joint3, z_joint3)
    phi_vert = math.atan2(h, r)
    cos_alpha = (l1 ** 2 + d ** 2 - l2 ** 2) / (2 * l1 * d)
    cos_alpha = max(min(cos_alpha, 1.0), -1.0)
    alpha = math.acos(cos_alpha)
    if r > 0:
        theta1 = phi_xz
        theta2_vert = phi_vert - alpha
    else:
        theta1 = phi_xz
        theta2_vert = phi_vert + alpha if h > 0 else phi_vert - alpha
    theta2_final = theta2_vert
    theta3 = 0.0
    return (theta1, theta2_final, theta3)