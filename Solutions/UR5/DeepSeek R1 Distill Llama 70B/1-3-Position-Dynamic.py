import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Calculates the joint angles needed to reach the desired TCP position.
    :param p: The desired TCP position as (x, y, z).
    :return: A tuple of joint angles (θ1, θ2, θ3) in radians.
    """
    x, y, z = p
    y_joint3 = y - 0.093
    theta1 = math.atan2(x, z)
    theta2 = math.atan2(y_joint3, z)
    theta3 = math.atan2(y_joint3, z)
    return (theta1, theta2, theta3)