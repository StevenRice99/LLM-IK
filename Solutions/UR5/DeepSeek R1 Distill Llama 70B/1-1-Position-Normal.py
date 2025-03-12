import math

def inverse_kinematics(p: tuple[float, float, float]) -> float:
    """
    Calculates the joint angle needed to reach the given TCP position.
    :param p: The target position [x, y, z].
    :return: The joint angle θ in radians.
    """
    px, py, pz = p
    theta = math.atan2(px, pz)
    return theta