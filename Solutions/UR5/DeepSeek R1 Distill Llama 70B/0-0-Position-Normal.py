import math

def inverse_kinematics(p: tuple[float, float, float]) -> float:
    """
    Computes the joint angle needed to reach the given position.
    
    Args:
        p: A tuple containing the target position [x, y, z].
        
    Returns:
        The joint angle in radians needed to reach the target position.
    """
    x_target, y_target, _ = p
    theta = math.atan2(x_target, y_target)
    theta = -theta
    return theta