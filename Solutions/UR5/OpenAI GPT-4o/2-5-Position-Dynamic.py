import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    d1 = 0.39225
    d2 = 0.093
    d3 = 0.09465
    d4 = 0.0823
    θ1 = math.atan2(x, z)
    y_eff = y - d2
    z_eff = z - d1
    r = math.sqrt(y_eff ** 2 + z_eff ** 2)
    cosθ3 = (r ** 2 - d3 ** 2 - d4 ** 2) / (2 * d3 * d4)
    cosθ3 = max(-1, min(1, cosθ3))
    θ3 = math.acos(cosθ3)
    θ2 = math.atan2(y_eff, z_eff) - math.atan2(d4 * math.sin(θ3), d3 + d4 * math.cos(θ3))
    θ4 = 0
    return (θ1, θ2, θ3, θ4)