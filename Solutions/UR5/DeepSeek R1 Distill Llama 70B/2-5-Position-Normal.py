def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    theta1 = 0.0
    theta2 = 0.0
    theta3 = 0.0
    theta4 = 0.0
    return (theta1, theta2, theta3, theta4)