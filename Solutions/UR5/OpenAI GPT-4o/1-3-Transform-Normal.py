import math

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [x, y, z].
    :return: A list of the values to set the links to for reaching position "p" and orientation "r".
    """
    L1 = 0.425
    L2 = 0.39225
    L3 = 0.093
    px, py, pz = p
    θ1 = 0
    pz_adjusted = pz - L3
    D = (pz_adjusted ** 2 + px ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
    θ3 = math.atan2(math.sqrt(1 - D ** 2), D)
    θ2 = math.atan2(pz_adjusted, px) - math.atan2(L2 * math.sin(θ3), L1 + L2 * math.cos(θ3))
    return (θ1, θ2, θ3)