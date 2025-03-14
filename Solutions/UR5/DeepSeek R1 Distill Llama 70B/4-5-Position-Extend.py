import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.09465
    B = 0.0823
    cos_phi = (z - A) / B
    phi = math.acos(cos_phi)
    sin_phi = math.sqrt(1 - cos_phi ** 2)
    adjusted_x = x - B * sin_phi
    adjusted_y = y - B * cos_phi
    theta = math.atan2(adjusted_y, adjusted_x)
    return (theta, phi)