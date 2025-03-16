import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p".
    """
    x, y, z = p
    theta1 = math.atan2(y, x)
    L1 = 0.13585
    L2 = 0.425
    L3 = 0.39225
    r = math.sqrt(x ** 2 + y ** 2)
    z_prime = z - L1
    D = math.sqrt(r ** 2 + z_prime ** 2)
    cos_theta3 = (D ** 2 - L2 ** 2 - L3 ** 2) / (2 * L2 * L3)
    cos_theta3 = max(-1.0, min(1.0, cos_theta3))
    theta3 = math.acos(cos_theta3)
    alpha = math.atan2(z_prime, r)
    beta = math.atan2(L3 * math.sin(theta3), L2 + L3 * math.cos(theta3))
    theta2_elbow_up = alpha - beta
    theta2_elbow_down = alpha + beta
    theta2 = theta2_elbow_up
    return (theta1, theta2, theta3)