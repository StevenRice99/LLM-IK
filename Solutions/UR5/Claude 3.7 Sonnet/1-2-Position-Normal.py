def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    l1 = 0.425
    l2 = 0.39225
    r = np.sqrt(x ** 2 + z ** 2)
    cos_theta2 = (r ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2)
    cos_theta2 = np.clip(cos_theta2, -1.0, 1.0)
    theta2_elbow_down = np.arccos(cos_theta2)
    theta2_elbow_up = -np.arccos(cos_theta2)
    phi = np.arctan2(z, x)
    psi_down = np.arctan2(l2 * np.sin(theta2_elbow_down), l1 + l2 * np.cos(theta2_elbow_down))
    theta1_elbow_down = phi - psi_down
    psi_up = np.arctan2(l2 * np.sin(theta2_elbow_up), l1 + l2 * np.cos(theta2_elbow_up))
    theta1_elbow_up = phi - psi_up
    x_down = l1 * np.cos(theta1_elbow_down) + l2 * np.cos(theta1_elbow_down + theta2_elbow_down)
    z_down = l1 * np.sin(theta1_elbow_down) + l2 * np.sin(theta1_elbow_down + theta2_elbow_down)
    error_down = (x - x_down) ** 2 + (z - z_down) ** 2
    x_up = l1 * np.cos(theta1_elbow_up) + l2 * np.cos(theta1_elbow_up + theta2_elbow_up)
    z_up = l1 * np.sin(theta1_elbow_up) + l2 * np.sin(theta1_elbow_up + theta2_elbow_up)
    error_up = (x - x_up) ** 2 + (z - z_up) ** 2
    if error_down <= error_up:
        return (theta1_elbow_down, theta2_elbow_down)
    else:
        return (theta1_elbow_up, theta2_elbow_up)