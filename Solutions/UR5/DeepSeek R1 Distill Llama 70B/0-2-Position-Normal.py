import math

def inverse_kinematics(p: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Computes the joint angles needed to reach the given position p.
    
    Args:
        p: A tuple (x, y, z) representing the target position.
        
    Returns:
        A tuple (q1, q2, q3) of joint angles in radians.
    """
    A = 0.13585
    B = -0.1197
    p_x, p_y, p_z = p
    mag_sq = p_x ** 2 + p_y ** 2
    numerator = mag_sq - (A ** 2 + B ** 2)
    denominator = 2 * A * B
    cos_q2 = numerator / denominator
    cos_q2 = max(min(cos_q2, 1.0), -1.0)
    q2 = math.acos(cos_q2)
    target_angle = math.atan2(p_y, p_x)
    angle_offset = math.atan2(B, A)
    q1 = target_angle - angle_offset
    q3_val = (p_z - 0.425) / 0.39225
    q3_val = max(min(q3_val, 1.0), -1.0)
    q3 = math.asin(q3_val)
    return (q1, q2, q3)