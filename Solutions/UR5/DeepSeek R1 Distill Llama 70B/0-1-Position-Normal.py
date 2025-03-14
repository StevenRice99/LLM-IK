import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    theta2 = math.acos(z / 0.425)
    sin_theta2 = math.sin(theta2)
    cos_theta2 = math.cos(theta2)
    K = 0.425 * sin_theta2
    L = 0.01615
    denominator = K ** 2 + L ** 2
    cos_theta1 = (K * x + L * y) / denominator
    sin_theta1 = (-L * x + K * y) / denominator
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)