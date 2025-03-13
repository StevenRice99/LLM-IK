import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    Args:
        p: A tuple containing the desired TCP position [x, y, z].
    
    Returns:
        A tuple of three floats representing the joint angles θ1, θ2, θ3 in radians.
    """
    x_d, y_d, z_d = p
    theta2 = math.acos(y_d / 0.093)
    theta1 = 0.0
    theta3 = math.atan2(x_d, z_d)
    return (theta1, theta2, theta3)