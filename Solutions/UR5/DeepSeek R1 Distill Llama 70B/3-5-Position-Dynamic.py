import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint angles needed to reach the given position p for a 3-DOF serial manipulator.
    
    :param p: The target position (x, y, z) for the TCP.
    :return: A tuple of three floats representing the joint angles theta1, theta2, and theta3 in radians.
    """
    px, py, pz = p
    theta1 = math.atan2(-px, py)
    theta2 = 0.0
    numerator = pz - 0.09465 * math.cos(theta1)
    denominator = 0.0823
    if abs(numerator / denominator) > 1:
        adjusted_numerator = numerator
        while abs(adjusted_numerator / denominator) > 1:
            adjusted_numerator -= math.copysign(denominator * 2 * math.pi, numerator)
        theta3 = math.asin(adjusted_numerator / denominator) - theta1
    else:
        theta3 = math.asin(numerator / denominator) - theta1
    return (theta1, theta2, theta3)