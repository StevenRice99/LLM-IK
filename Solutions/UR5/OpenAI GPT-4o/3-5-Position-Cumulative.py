import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    px, py, pz = p
    d1 = 0.093
    d2 = 0.09465
    d3 = 0.0823
    theta1 = math.atan2(px, pz)
    y_prime = py - d1 - d3
    z_prime = pz - d2
    theta2 = math.atan2(y_prime, z_prime)
    theta3 = 0
    return (theta1, theta2, theta3)