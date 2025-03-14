import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    l1 = math.sqrt(0.1197 ** 2 + 0.425 ** 2)
    l2 = 0.39225
    d = math.sqrt(x ** 2 + z ** 2)
    cosθ2 = (l1 ** 2 + l2 ** 2 - d ** 2) / (2 * l1 * l2)
    cosθ2 = max(min(cosθ2, 1.0), -1.0)
    θ2 = math.acos(cosθ2)
    θ1 = math.atan2(x, z) - θ2
    return (θ1, θ2)