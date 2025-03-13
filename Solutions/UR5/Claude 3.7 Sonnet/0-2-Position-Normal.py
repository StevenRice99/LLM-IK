def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    theta1 = np.arctan2(y, x)
    x_prime = x * np.cos(theta1) + y * np.sin(theta1)
    y_prime = -x * np.sin(theta1) + y * np.cos(theta1)
    z_prime = z
    l1 = 0.13585
    l2 = np.sqrt(0.1197 ** 2 + 0.425 ** 2)
    l3 = 0.39225
    alpha = np.arctan2(0.425, 0.1197)
    r = np.sqrt(x_prime ** 2 + (z_prime - l1) ** 2)
    cos_theta3 = (r ** 2 - l2 ** 2 - l3 ** 2) / (2 * l2 * l3)
    cos_theta3 = np.clip(cos_theta3, -1.0, 1.0)
    theta3 = np.arccos(cos_theta3)
    beta = np.arctan2(z_prime - l1, x_prime)
    gamma = np.arcsin(l3 * np.sin(theta3) / r)
    theta2 = beta - gamma
    theta2 = theta2 - alpha
    theta3 = np.pi - theta3
    return (theta1, theta2, theta3)