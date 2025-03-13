import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    l1 = 0.39225
    l2 = 0.093
    l3 = 0.09465
    l4 = 0.0823
    theta4 = 0.0
    theta1 = math.atan2(x, z)
    r = math.sqrt(x ** 2 + z ** 2)
    wrist_y = y - l4
    j2_y = 0
    j2_z = l1
    dr = r
    dy = wrist_y - j2_y
    dz = 0 - j2_z
    theta2 = math.atan2(dr, dz)
    j3_r = l1 * math.sin(theta2)
    j3_y = l2
    j3_z = l1 * math.cos(theta2)
    dr_j3 = r - j3_r
    dy_j3 = wrist_y - j3_y
    dz_j3 = 0 - j3_z
    d_j3 = math.sqrt(dr_j3 ** 2 + dy_j3 ** 2 + dz_j3 ** 2)
    theta3 = math.atan2(dy_j3, math.sqrt(dr_j3 ** 2 + dz_j3 ** 2))
    return (theta1, theta2, theta3, theta4)