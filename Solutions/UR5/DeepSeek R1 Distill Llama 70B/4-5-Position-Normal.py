import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    Args:
        p: A tuple containing the desired TCP position [x, y, z].
    
    Returns:
        A tuple of two floats representing the joint angles θ1 and θ2 in radians.
    """
    x_d, y_d, z_d = p
    radius = 0.0823
    if not (abs(x_d) < radius * 2 and abs(y_d) < radius * 2):
        raise ValueError('Target position is out of reach.')
    theta1 = math.atan2(-x_d, y_d)
    theta2 = 0.0
    return (theta1, theta2)