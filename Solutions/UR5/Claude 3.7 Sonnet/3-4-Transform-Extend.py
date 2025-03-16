def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
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
    joint1 = math.atan2(px, pz)

    def euler_to_rotation_matrix(euler_angles):
        x, y, z = euler_angles
        Rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
        Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
        Rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
        return np.matmul(np.matmul(Rz, Ry), Rx)
    target_matrix = euler_to_rotation_matrix([rx, ry, rz])
    joint1_matrix = np.array([[math.cos(joint1), 0, math.sin(joint1)], [0, 1, 0], [-math.sin(joint1), 0, math.cos(joint1)]])
    joint1_inv = np.transpose(joint1_matrix)
    joint2_matrix = np.matmul(joint1_inv, target_matrix)
    joint2 = math.atan2(joint2_matrix[1, 0], joint2_matrix[0, 0])
    return (joint1, joint2)