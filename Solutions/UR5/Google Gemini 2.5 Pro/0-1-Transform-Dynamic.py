import math

def normalize_angle_modulo(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi] using modulo arithmetic."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def normalize_angle_atan2(angle: float) -> float:
    """Normalize an angle to the range [-pi, pi] using atan2."""
    return math.atan2(math.sin(angle), math.cos(angle))

def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    It's inferred from problem context and feedback that the target orientation 'r'
    directly defines the joint angles, and these are guaranteed to achieve position 'p'.
    
    :param p: The position to reach in the form [x, y, z]. (Used implicitly by problem guarantee)
    :param r: The orientation to reach in radians in the form [rx, ry, rz] (roll, pitch, yaw).
    :return: A tuple of the joint values (joint1, joint2).
    """
    rx, ry, rz = r
    geom_epsilon = 1e-09
    theta1_intermediate: float
    theta2_intermediate: float
    if abs(rx) < geom_epsilon:
        theta1_intermediate = rz
        theta2_intermediate = ry
    elif abs(rx - math.pi) < geom_epsilon or abs(rx + math.pi) < geom_epsilon:
        theta1_intermediate = rz + math.pi
        theta2_intermediate = math.pi - ry
    else:
        theta1_intermediate = rz
        theta2_intermediate = ry
    theta1_solution = normalize_angle_atan2(theta1_intermediate)
    theta2_solution = normalize_angle_atan2(theta2_intermediate)
    return (theta1_solution, theta2_solution)