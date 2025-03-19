import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1_y = 0.13585
    L2_y = -0.1197
    L2_z = 0.425
    L3_z = 0.39225
    L4_y = 0.093
    L5_z = 0.09465
    theta1 = math.atan2(x, y)
    z5 = z - L5_z
    x4 = x - L4_y * math.sin(theta1)
    y4 = y - L4_y * math.cos(theta1)
    z4 = z5
    r4 = math.sqrt(x4 ** 2 + y4 ** 2)
    r2 = r4 - L1_y
    L2 = math.sqrt(L2_y ** 2 + L2_z ** 2)
    L3 = L3_z
    alpha2 = math.atan2(L2_z, -L2_y)
    d = math.sqrt(r2 ** 2 + z4 ** 2)
    cos_theta3 = (d ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    theta3_internal = math.acos(cos_theta3)
    phi = math.atan2(z4, r2)
    theta3_up = theta3_internal
    psi_up = math.atan2(L3 * math.sin(theta3_up), L2 + L3 * math.cos(theta3_up))
    theta2_up = phi - psi_up
    theta3_down = -theta3_internal
    psi_down = math.atan2(L3 * math.sin(theta3_down), L2 + L3 * math.cos(theta3_down))
    theta2_down = phi - psi_down
    x4_up = L1_y + (L2 * math.cos(theta2_up) + L3 * math.cos(theta2_up + theta3_up)) * math.cos(theta1)
    y4_up = L1_y + (L2 * math.cos(theta2_up) + L3 * math.cos(theta2_up + theta3_up)) * math.sin(theta1)
    z4_up = L2 * math.sin(theta2_up) + L3 * math.sin(theta2_up + theta3_up)
    x4_down = L1_y + (L2 * math.cos(theta2_down) + L3 * math.cos(theta2_down + theta3_down)) * math.cos(theta1)
    y4_down = L1_y + (L2 * math.cos(theta2_down) + L3 * math.cos(theta2_down + theta3_down)) * math.sin(theta1)
    z4_down = L2 * math.sin(theta2_down) + L3 * math.sin(theta2_down + theta3_down)
    error_up = (x4_up - x4) ** 2 + (y4_up - y4) ** 2 + (z4_up - z4) ** 2
    error_down = (x4_down - x4) ** 2 + (y4_down - y4) ** 2 + (z4_down - z4) ** 2
    if error_up <= error_down:
        theta2 = theta2_up
        theta3 = theta3_up
    else:
        theta2 = theta2_down
        theta3 = theta3_down
    theta4 = -(theta2 + theta3)
    theta5 = 0
    return (theta1, theta2, theta3, theta4, theta5)