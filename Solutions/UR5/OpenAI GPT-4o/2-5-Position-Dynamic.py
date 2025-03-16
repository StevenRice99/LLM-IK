import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float, float]:
    """
    Gets the joint values needed to reach position "p".
    :param p: The position to reach in the form [x, y, z].
    :return: A tuple of the values to set the joints to for reaching position "p".
    """
    x, y, z = p
    L1 = 0.39225
    L2 = 0.093
    L3 = 0.09465
    L4 = 0.0823
    r = math.sqrt(x ** 2 + z ** 2)
    θ1 = math.atan2(x, z)
    h = y - L2
    d = math.sqrt(r ** 2 + h ** 2)
    θ2 = math.atan2(h, r) - math.acos((L3 ** 2 + d ** 2 - L4 ** 2) / (2 * L3 * d))
    θ3 = math.acos((L3 ** 2 + L4 ** 2 - d ** 2) / (2 * L3 * L4))
    θ4 = math.pi / 2 - θ2 - θ3
    return (θ1, θ2, θ3, θ4)