import numpy as np

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.425
    L2 = 0.39225
    r = np.sqrt(x ** 2 + z ** 2)
    if r > L1 + L2 or r < abs(L1 - L2):
        raise ValueError('Target position is out of reach')
    D = (r ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    theta2 = np.arccos(D)
    theta1_up = np.arctan2(z, x) - np.arctan2(L2 * np.sin(theta2), L1 + L2 * np.cos(theta2))
    theta2_down = -theta2
    theta1_down = np.arctan2(z, x) - np.arctan2(L2 * np.sin(theta2_down), L1 + L2 * np.cos(theta2_down))

    def forward_kinematics(theta1, theta2):
        x_pos = L1 * np.cos(theta1) + L2 * np.cos(theta1 + theta2)
        z_pos = L1 * np.sin(theta1) + L2 * np.sin(theta1 + theta2)
        return (x_pos, z_pos)
    x_up, z_up = forward_kinematics(theta1_up, theta2)
    x_down, z_down = forward_kinematics(theta1_down, theta2_down)
    dist_up = np.sqrt((x_up - x) ** 2 + (z_up - z) ** 2)
    dist_down = np.sqrt((x_down - x) ** 2 + (z_down - z) ** 2)
    if dist_up < dist_down:
        return (theta1_up, theta2)
    else:
        return (theta1_down, theta2_down)