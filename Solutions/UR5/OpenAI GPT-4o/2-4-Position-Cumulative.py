import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.39225
    d2 = 0.093
    d3 = 0.09465
    z_adjusted = z - d3
    θ1 = math.atan2(x, z_adjusted)
    r = math.sqrt(x ** 2 + z_adjusted ** 2)
    h = z_adjusted - d1
    L = math.sqrt(r ** 2 + h ** 2)
    cos_θ2 = (d2 ** 2 + L ** 2 - d3 ** 2) / (2 * d2 * L)
    cos_θ2 = max(min(cos_θ2, 1.0), -1.0)
    sin_θ2 = math.sqrt(1 - cos_θ2 ** 2)
    θ2 = math.atan2(h, r) - math.atan2(sin_θ2, cos_θ2)
    cos_θ3 = (d2 ** 2 + d3 ** 2 - L ** 2) / (2 * d2 * d3)
    cos_θ3 = max(min(cos_θ3, 1.0), -1.0)
    sin_θ3 = math.sqrt(1 - cos_θ3 ** 2)
    θ3 = math.atan2(sin_θ3, cos_θ3)
    return (θ1, θ2, θ3)