def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    joint1 = np.arctan2(x, z)
    tcp_length = 0.09465
    xz_distance = np.sqrt(x ** 2 + z ** 2)
    if abs(xz_distance - tcp_length) > 1e-06:
        scale = tcp_length / max(xz_distance, 1e-10)
        joint1 = np.arctan2(x * scale, z * scale)

    def rot_y(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    def rot_z(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def rot_x(angle):
        c = np.cos(angle)
        s = np.sin(angle)
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    R_target = rot_x(rx) @ rot_y(ry) @ rot_z(rz)
    R_joint1 = rot_y(joint1)
    R_joint2_needed = R_joint1.T @ R_target
    joint2 = np.arctan2(R_joint2_needed[1, 0], R_joint2_needed[0, 0])
    return (joint1, joint2)