import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    px, py, pz = p
    A = 0.09465
    B = 0.0823
    distance = math.sqrt(px ** 2 + py ** 2)
    cos_theta2 = (px ** 2 + py ** 2 - A ** 2 - B ** 2) / (2 * A * B)
    theta2 = math.acos(cos_theta2)
    theta1 = math.atan2(px, -py) - theta2
    return (theta1, theta2)