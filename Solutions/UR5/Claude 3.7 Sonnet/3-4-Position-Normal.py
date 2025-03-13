def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    import numpy as np
    x, y, z = p
    joint1 = np.arctan2(x, z)
    joint2 = 0.0
    tcp_length = 0.09465
    xz_distance = np.sqrt(x ** 2 + z ** 2)
    if abs(xz_distance - tcp_length) > 1e-06:
        scale = tcp_length / max(xz_distance, 1e-10)
        joint1 = np.arctan2(x * scale, z * scale)
    return (joint1, joint2)