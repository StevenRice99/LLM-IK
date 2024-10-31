def inverse_kinematics(p: list) -> list[float]:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    import math

    x, y, z = p
    # Link lengths
    L1 = 0.4  # Length from Joint 2 to Joint 3
    L2 = 0.4  # Length from Joint 3 to End Effector
    d1 = 0.4  # Base height (from base to Joint 2)

    # Compute theta1
    theta1 = math.atan2(y, x)
    # Compute r and s
    r = math.hypot(x, y)
    s = z - d1
    # Check reachability
    D = (r**2 + s**2 - L1**2 - L2**2) / (2 * L1 * L2)
    # Clamp D to [-1, 1]
    if D < -1 or D > 1:
        raise ValueError("Position is unreachable.")
    theta3 = math.acos(D)
    # Compute theta2
    phi = math.atan2(s, r)
    psi = math.atan2(L2 * math.sin(theta3), L1 + L2 * math.cos(theta3))
    theta2 = phi - psi
    return [theta1, theta2, theta3]
