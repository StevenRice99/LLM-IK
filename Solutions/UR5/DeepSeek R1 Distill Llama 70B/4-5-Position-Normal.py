import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    :param p: The target position (x, y, z) for the TCP.
    :return: A tuple of two floats representing the joint angles θ1 and θ2 in radians.
    """
    px, py, pz = p
    A = 0.0823
    theta2 = 0.0
    theta1 = math.atan2(-px, py)
    return (theta1, theta2)