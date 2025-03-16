def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    px, py, pz = p
    rx, ry, rz = r
    L1 = 0.093
    L2 = 0.09465
    L3 = 0.0823
    theta3 = rx
    theta2 = rz - 1.570796325
    cos_theta2 = math.cos(theta2)
    sin_theta2 = math.sin(theta2)
    local_offset = [0, L3, 0]
    rotated_offset_x = -local_offset[1] * sin_theta2
    rotated_offset_y = local_offset[1] * cos_theta2
    j3_x = px - rotated_offset_x
    j3_y = py - rotated_offset_y
    j3_z = pz
    j2_x = 0
    j2_y = L1
    j2_z = 0
    v_x = j3_x - j2_x
    v_z = j3_z - j2_z
    theta1 = math.atan2(v_x, v_z)
    return (theta1, theta2, theta3)