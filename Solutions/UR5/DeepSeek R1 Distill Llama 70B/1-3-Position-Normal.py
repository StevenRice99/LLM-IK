import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    A = 0.425
    B = 0.39225
    C = 2 * A * x
    D = -2 * A * z
    E = B ** 2 - x ** 2 - z ** 2 - A ** 2
    magnitude = math.sqrt(C ** 2 + D ** 2)
    if magnitude == 0:
        theta1 = 0.0
    else:
        theta1 = math.atan2(C, D) - math.acos(E / magnitude)
    sin_phi = (-x - A * math.sin(theta1)) / B
    cos_phi = (z - A * math.cos(theta1)) / B
    if abs(sin_phi) > 1:
        sin_phi = math.copysign(1, sin_phi)
    if abs(cos_phi) > 1:
        cos_phi = math.copysign(1, cos_phi)
    phi = math.atan2(sin_phi, cos_phi)
    theta2 = phi - theta1
    theta3 = 0.0
    return (theta1, theta2, theta3)