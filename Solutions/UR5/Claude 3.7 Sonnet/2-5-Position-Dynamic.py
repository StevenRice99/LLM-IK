import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta4 = 0.0
    wrist_x = x
    wrist_y = y - 0.0823
    wrist_z = z
    theta1 = math.atan2(wrist_x, wrist_z)
    r_xz = math.sqrt(wrist_x ** 2 + wrist_z ** 2)
    j2_x = 0.39225 * math.sin(theta1)
    j2_z = 0.39225 * math.cos(theta1)
    dx = wrist_x - j2_x
    dy = wrist_y - 0
    dz = wrist_z - j2_z
    r_proj = math.sqrt(dx ** 2 + dz ** 2)
    distance = math.sqrt(r_proj ** 2 + dy ** 2)
    link2_length = 0.093
    link3_length = 0.09465
    cos_theta3 = (link2_length ** 2 + link3_length ** 2 - distance ** 2) / (2 * link2_length * link3_length)
    theta3 = math.acos(max(min(cos_theta3, 1.0), -1.0))
    beta = math.atan2(dy, r_proj)
    gamma = math.acos((link2_length ** 2 + distance ** 2 - link3_length ** 2) / (2 * link2_length * distance))
    theta2 = beta + gamma
    if wrist_y > 0:
        if wrist_z > 0:
            theta2 = -theta2
            theta3 = -theta3
        else:
            theta2 = theta2
            theta3 = -theta3
    elif wrist_z > 0:
        theta2 = theta2
        theta3 = -theta3
    else:
        theta2 = -theta2
        theta3 = -theta3
    return (theta1, theta2, theta3, theta4)