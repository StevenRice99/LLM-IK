import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.13585
    d2 = 0.1197
    d3 = 0.425
    d4 = 0.39225
    d5 = 0.093
    tcp_offset = 0.09465
    z -= tcp_offset
    theta1 = np.arctan2(y, x)
    r = np.sqrt(x ** 2 + y ** 2)
    y_proj = r - d5
    z_proj = z - d1
    L = np.sqrt(y_proj ** 2 + z_proj ** 2)
    cos_theta3 = (L ** 2 - d3 ** 2 - d4 ** 2) / (2 * d3 * d4)
    theta3 = np.arccos(np.clip(cos_theta3, -1.0, 1.0))
    alpha = np.arctan2(z_proj, y_proj)
    beta = np.arccos(np.clip((L ** 2 + d3 ** 2 - d4 ** 2) / (2 * L * d3), -1.0, 1.0))
    theta2 = alpha - beta
    theta4 = np.pi - theta3 - theta2
    theta5 = 0.0
    theta2 = -theta2
    theta3 = -theta3
    theta4 = -theta4
    return (theta1, theta2, theta3, theta4, theta5)