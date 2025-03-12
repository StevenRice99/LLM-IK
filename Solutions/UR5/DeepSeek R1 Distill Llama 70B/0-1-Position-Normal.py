import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the joint angles theta1 and theta2 to reach the given position p.
    
    Args:
        p: A tuple (x, y, z) representing the target position.
    
    Returns:
        A tuple (theta1, theta2) in radians.
    """
    x, y, z = p
    z_max = 0.425
    cos_theta2 = z / z_max
    theta2 = math.acos(cos_theta2)
    sin_theta2 = math.sqrt(1 - cos_theta2 ** 2)
    E = 0.425 * sin_theta2
    denominator = E ** 2 + 0.01615 ** 2
    cos_theta1 = (E * x + 0.01615 * y) / denominator
    sin_theta1 = (E * y - 0.01615 * x) / denominator
    norm = math.sqrt(cos_theta1 ** 2 + sin_theta1 ** 2)
    cos_theta1 /= norm
    sin_theta1 /= norm
    theta1 = math.atan2(sin_theta1, cos_theta1)
    return (theta1, theta2)