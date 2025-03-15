import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    θ1 = math.atan2(x, z)
    d = math.sqrt(x ** 2 + z ** 2)
    h = y - 0.39225
    L1 = 0.093
    L2 = 0.09465
    cosθ2 = (d ** 2 + h ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    cosθ2 = max(min(cosθ2, 1.0), -1.0)
    θ2 = math.acos(cosθ2)
    θ3 = math.atan2(y, x)
    θ4 = math.atan2(-z, x)
    return (θ1, θ2, θ3, θ4)