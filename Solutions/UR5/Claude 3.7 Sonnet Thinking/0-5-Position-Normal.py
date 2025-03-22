def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    d1 = 0.13585
    a1 = 0
    a2 = 0.425
    d3 = -0.1197
    a3 = 0.39225
    d5 = 0.093
    d6 = 0.09465
    dt = 0.0823
    theta1 = np.arctan2(y, x)
    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    wc_x = x - (d5 + dt) * s1
    wc_y = y + (d5 + dt) * c1
    wc_z = z - d6
    r = np.sqrt(wc_x ** 2 + wc_z ** 2)
    s = wc_z - d1
    side_a = a3
    side_b = np.sqrt(r ** 2 + (wc_y - d3) ** 2)
    side_c = a2
    angle_a = np.arccos((side_b ** 2 + side_c ** 2 - side_a ** 2) / (2 * side_b * side_c))
    angle_b = np.arccos((side_a ** 2 + side_c ** 2 - side_b ** 2) / (2 * side_a * side_c))
    angle_r = np.arctan2(wc_y - d3, r)
    theta2 = np.pi / 2 - (angle_a + angle_r)
    theta3 = np.pi / 2 - angle_b
    theta4 = 0
    theta5 = 0
    theta6 = 0
    return (theta1, theta2, theta3, theta4, theta5, theta6)