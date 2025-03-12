import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint angles needed to reach the given position using an accurate kinematic model.
    
    Args:
        p: A tuple containing the target position [x, y, z].
        
    Returns:
        A tuple of the joint angles [joint1, joint2, joint3] needed to reach the target position.
    """
    x_target, y_target, z_target = p
    joint1 = math.atan2(y_target, x_target)
    adjusted_x = x_target * math.cos(joint1) + y_target * math.sin(joint1)
    adjusted_y = -x_target * math.sin(joint1) + y_target * math.cos(joint1)
    adjusted_z = z_target
    joint3 = math.atan2(adjusted_x, adjusted_z)
    joint2 = math.atan2(adjusted_y, adjusted_z)
    return (joint1, joint2, joint3)