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
    l1 = 0.425
    l2 = 0.39225
    r_target = np.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (r_target ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2_elbow_up = -np.arccos(cos_theta2)
    theta2_elbow_down = np.arccos(cos_theta2)
    phi = np.arctan2(x, z)
    psi_up = np.arctan2(l2 * np.sin(theta2_elbow_up), l1 + l2 * np.cos(theta2_elbow_up))
    theta1_elbow_up = phi - psi_up
    psi_down = np.arctan2(l2 * np.sin(theta2_elbow_down), l1 + l2 * np.cos(theta2_elbow_down))
    theta1_elbow_down = phi - psi_down
    ry_normalized = (ry + np.pi) % (2 * np.pi) - np.pi
    if ry_normalized < 0:
        theta1 = theta1_elbow_up
        theta2 = theta2_elbow_up
    else:
        theta1 = theta1_elbow_down
        theta2 = theta2_elbow_down
    if abs(rx - np.pi) < 0.1:
        if theta2 > 0:
            theta2 = -theta2
        else:
            theta2 = -theta2
        if theta1 > 0:
            theta1 = theta1 - np.pi
        else:
            theta1 = theta1 + np.pi
    theta1 = (theta1 + np.pi) % (2 * np.pi) - np.pi
    theta2 = (theta2 + np.pi) % (2 * np.pi) - np.pi
    return (theta1, theta2)