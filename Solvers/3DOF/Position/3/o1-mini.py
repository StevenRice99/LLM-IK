import math

def inverse_kinematics(p: list) -> list[float]:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    x, y, z = p
    l0 = 0.4  # Offset from Joint 1 to Joint 2 along Z-axis
    l1 = 0.4  # Length from Joint 2 to Joint 3
    l2 = 0.4  # Length from Joint 3 to End Effector

    # Compute theta1
    theta1 = -math.atan2(x, y)

    # Compute the radial distance in the XY-plane
    r = math.sqrt(x**2 + y**2)

    # Effective Z after accounting for the end effector's offset
    z_eff = z - l0

    # Compute cos(theta3) using the law of cosines
    cos_theta3 = (r**2 + z_eff**2 - l1**2 - l2**2) / (2 * l1 * l2)
    # Clamp cos_theta3 to the valid range to avoid numerical issues
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    # Compute theta3 (elbow-down configuration)
    theta3 = -math.acos(cos_theta3)

    # Compute sin(theta3) for theta2 calculation
    sin_theta3 = math.sin(theta3)

    # Compute intermediate variables for theta2
    k1 = l1 + l2 * math.cos(theta3)
    k2 = l2 * sin_theta3

    # Compute theta2
    theta2 = math.atan2(z_eff, r) - math.atan2(k2, k1) - (math.pi / 2)

    return [theta1, theta2, theta3]
