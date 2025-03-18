import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the links to for reaching position "p".
    """
    px, py, pz = p
    theta1 = math.atan2(px, pz)
    py_eff = py - 0.093
    pz_eff = pz - 0.09465
    l2 = 0.093
    d_eff = math.sqrt(py_eff ** 2 + pz_eff ** 2)
    cos_theta2 = py_eff / d_eff
    theta2 = math.asin(cos_theta2)
    theta3 = 0
    return (theta1, theta2, theta3)