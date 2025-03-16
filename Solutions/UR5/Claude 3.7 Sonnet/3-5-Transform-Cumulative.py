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
    tcp_offset = 1.570796325

    def euler_to_rotation_matrix(euler_angles):
        x, y, z = euler_angles
        Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        return np.matmul(np.matmul(Rz, Ry), Rx)
    target_matrix = euler_to_rotation_matrix([rx, ry, rz])
    offset_matrix = euler_to_rotation_matrix([0, 0, tcp_offset])
    offset_inv = np.transpose(offset_matrix)
    adjusted_target = np.matmul(target_matrix, offset_inv)
    joint1 = math.atan2(px, pz)
    joint1_matrix = np.array([[math.cos(joint1), 0, math.sin(joint1)], [0, 1, 0], [-math.sin(joint1), 0, math.cos(joint1)]])
    joint1_inv = np.transpose(joint1_matrix)
    local_pos = np.matmul(joint1_inv, np.array([px, py, pz]))
    local_x, local_y, local_z = local_pos
    target_local_y = local_y - L1
    dist_sq = local_x ** 2 + target_local_y ** 2 + local_z ** 2
    dist = math.sqrt(dist_sq)
    if dist > L2 + L3 or dist < abs(L2 - L3):
        if dist > L2 + L3:
            joint3 = 0
        else:
            joint3 = math.pi
    else:
        cos_joint3 = (L2 ** 2 + L3 ** 2 - dist_sq) / (2 * L2 * L3)
        cos_joint3 = max(min(cos_joint3, 1.0), -1.0)
        joint3 = math.pi - math.acos(cos_joint3)
    L3_x = L3 * math.sin(joint3)
    L3_y = L3 * math.cos(joint3)
    joint2 = math.atan2(local_x, target_local_y - L3_y)
    joint2_matrix = np.array([[math.cos(joint2), -math.sin(joint2), 0], [math.sin(joint2), math.cos(joint2), 0], [0, 0, 1]])
    R12 = np.matmul(joint1_matrix, joint2_matrix)
    R12_inv = np.transpose(R12)
    R3_needed = np.matmul(R12_inv, adjusted_target)
    return (joint1, joint2, joint3)