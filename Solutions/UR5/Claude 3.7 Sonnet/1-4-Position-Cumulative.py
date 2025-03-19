import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    L3_y = 0.093
    L4_z = 0.09465
    y_offset = -0.1197
    r_xz = math.sqrt(x ** 2 + z ** 2)
    if r_xz < 1e-06:
        theta1 = 0.0
    else:
        theta1 = math.atan2(x, z)
    wrist_x = x - L4_z * math.sin(theta1)
    wrist_z = z - L4_z * math.cos(theta1)
    wrist_y = y
    elbow_x = wrist_x
    elbow_y = wrist_y - L3_y
    elbow_z = wrist_z
    shoulder_x = 0
    shoulder_y = y_offset
    shoulder_z = 0
    dx = elbow_x - shoulder_x
    dy = elbow_y - shoulder_y
    dz = elbow_z - shoulder_z
    d_xz = math.sqrt(dx ** 2 + dz ** 2)
    d = math.sqrt(d_xz ** 2 + dy ** 2)
    cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
    theta2 = math.acos(cos_theta2)
    phi = math.atan2(dy, d_xz)
    alpha = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
    theta3 = phi - alpha - theta2
    theta4 = 0.0
    target_local_x = x * math.cos(-theta1) - z * math.sin(-theta1)
    target_local_z = x * math.sin(-theta1) + z * math.cos(-theta1)
    if target_local_z < 0:
        if theta1 > 0:
            theta1 = theta1 - math.pi
        else:
            theta1 = theta1 + math.pi
        wrist_x = x - L4_z * math.sin(theta1)
        wrist_z = z - L4_z * math.cos(theta1)
        wrist_y = y
        elbow_x = wrist_x
        elbow_y = wrist_y - L3_y
        elbow_z = wrist_z
        dx = elbow_x - shoulder_x
        dy = elbow_y - shoulder_y
        dz = elbow_z - shoulder_z
        d_xz = math.sqrt(dx ** 2 + dz ** 2)
        d = math.sqrt(d_xz ** 2 + dy ** 2)
        cos_theta2 = (d ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
        cos_theta2 = max(min(cos_theta2, 1.0), -1.0)
        theta2 = -math.acos(cos_theta2)
        phi = math.atan2(dy, d_xz)
        alpha = math.atan2(L2 * math.sin(theta2), L1 + L2 * math.cos(theta2))
        theta3 = phi - alpha - theta2
    return (theta1, theta2, theta3, theta4)