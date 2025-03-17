import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    Args:
        p: A tuple (x, y, z) representing the target position.
    
    Returns:
        A tuple (θ1, θ2, θ3) of joint angles in radians.
    """
    x, y, z = p
    l1 = 0.13585
    l2 = 0.496
    l3 = 0.39225
    if x == 0:
        theta1 = 0.0
    else:
        theta1 = math.atan(y / x)
    target_yz = (y - l1 * math.sin(theta1), z)
    distance = math.sqrt(target_yz[0] ** 2 + target_yz[1] ** 2)
    if distance > l2 + l3:
        raise ValueError('Target position is unreachable.')
    cos_theta2 = (l2 ** 2 + distance ** 2 - l3 ** 2) / (2 * l2 * distance)
    theta2 = math.acos(cos_theta2)
    sin_theta3 = target_yz[1] / distance
    theta3 = math.asin(sin_theta3)
    return (theta1, theta2, theta3)