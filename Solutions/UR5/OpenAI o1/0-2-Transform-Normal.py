def inverse_kinematics(p: tuple[float, float, float], r: tuple[float, float, float]) -> tuple[float, float, float]:
    """
    Gets the joint values needed to reach position "p" and orientation "r".
    :param p: The position to reach in the form [x, y, z].
    :param r: The orientation to reach in radians in the form [roll, pitch, yaw].
    :return: A tuple of (θ1, θ2, θ3) in radians.
    """
    import math
    x, y, z = p
    roll, pitch, yaw = r
    theta1 = yaw
    p2_x = x * math.cos(theta1) + y * math.sin(theta1) - 0.13585
    p2_y = -x * math.sin(theta1) + y * math.cos(theta1)
    p2_z = z
    L2 = 0.425
    L3 = 0.39225
    rhs_sin = (p2_x - L3 * math.sin(pitch)) / L2
    rhs_cos = (p2_z - L3 * math.cos(pitch)) / L2
    theta2 = math.atan2(rhs_sin, rhs_cos)
    theta3 = pitch - theta2
    return (theta1, theta2, theta3)