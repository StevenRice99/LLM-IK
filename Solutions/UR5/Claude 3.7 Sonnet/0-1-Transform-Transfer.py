def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    import math
    import numpy as np
    x, y, z = p
    rx, ry, rz = r
    if abs(z) <= 0.425:
        theta2 = math.acos(z / 0.425)
    else:
        theta2 = 0 if z > 0 else math.pi
    sin_theta2 = math.sin(theta2)
    A = 0.425 * sin_theta2
    B = 0.1197

    def rot_x(angle):
        return np.array([[1, 0, 0], [0, math.cos(angle), -math.sin(angle)], [0, math.sin(angle), math.cos(angle)]])

    def rot_y(angle):
        return np.array([[math.cos(angle), 0, math.sin(angle)], [0, 1, 0], [-math.sin(angle), 0, math.cos(angle)]])

    def rot_z(angle):
        return np.array([[math.cos(angle), -math.sin(angle), 0], [math.sin(angle), math.cos(angle), 0], [0, 0, 1]])
    R_rpy = rot_z(rz) @ rot_y(ry) @ rot_x(rx)
    R_pos = rot_z(0) @ rot_y(theta2)
    R_neg = rot_z(0) @ rot_y(-theta2)
    y_axis_rpy = R_rpy[:, 1]
    y_axis_pos = R_pos[:, 1]
    y_axis_neg = R_neg[:, 1]
    dot_pos = np.abs(np.dot(y_axis_rpy, y_axis_pos))
    dot_neg = np.abs(np.dot(y_axis_rpy, y_axis_neg))
    if dot_neg > dot_pos:
        theta2 = -theta2
        A = 0.425 * math.sin(theta2)
    C = math.sqrt(A ** 2 + B ** 2)
    D = math.atan2(B, A)
    if C < 1e-10:
        theta1 = rz
    else:
        theta1_plus_D = math.atan2(y, x)
        theta1 = theta1_plus_D - D
    R_current = rot_z(theta1) @ rot_y(theta2)
    R_alt = rot_z(theta1 + math.pi) @ rot_y(theta2)
    diff_current = np.sum(np.abs(R_current - R_rpy))
    diff_alt = np.sum(np.abs(R_alt - R_rpy))
    if diff_alt < diff_current:
        theta1 += math.pi
    theta1 = (theta1 + math.pi) % (2 * math.pi) - math.pi
    theta2 = (theta2 + math.pi) % (2 * math.pi) - math.pi
    return (theta1, theta2)