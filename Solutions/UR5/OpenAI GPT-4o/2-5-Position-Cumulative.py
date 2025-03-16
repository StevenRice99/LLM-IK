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
    θ1 = math.atan2(x, z)
    z2 = z - L1
    r2 = math.sqrt(x ** 2 + z2 ** 2)
    θ2 = math.atan2(y, r2)
    y3 = y - L2
    r3 = math.sqrt(x ** 2 + y3 ** 2)
    θ3 = math.atan2(x, y3)
    z4 = z2 - L3
    r4 = math.sqrt(y3 ** 2 + z4 ** 2)
    θ4 = math.atan2(y3, z4)
    return (θ1, θ2, θ3, θ4)