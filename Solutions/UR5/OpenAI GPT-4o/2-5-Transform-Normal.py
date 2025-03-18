import numpy as np

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    x, y, z = p
    r_x, r_y, r_z = r
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    L4 = 0.0823
    theta1 = np.arctan2(y, x)
    wrist_x = x - L4 * np.cos(r_z) * np.cos(r_y)
    wrist_y = y - L4 * np.sin(r_z) * np.cos(r_y)
    wrist_z = z - L4 * np.sin(r_y)
    d = np.sqrt(wrist_x ** 2 + wrist_y ** 2)
    z_offset = wrist_z - L1
    theta2 = np.arctan2(z_offset, d)
    L23 = np.sqrt(L2 ** 2 + L3 ** 2)
    theta3 = np.arctan2(L3, L2)
    theta4 = r_y - theta2 - theta3
    return (theta1, theta2, theta3, theta4)