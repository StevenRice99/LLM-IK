import math
from typing import Tuple

def inverse_kinematics(p: Tuple[float, float, float], r: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """
    Solves for the joint angles θ1, θ2, θ3 to reach the desired position p and orientation r.
    
    Args:
        p: The desired position (x, y, z) of the TCP.
        r: The desired orientation (roll, pitch, yaw) of the TCP in radians.
    
    Returns:
        A tuple of joint angles (θ1, θ2, θ3) in radians.
    """
    theta1 = math.atan2(p[0], p[2])
    theta2 = math.atan2(p[1], p[0])
    theta3 = math.atan2(p[0], p[2])
    return (theta1, theta2, theta3)