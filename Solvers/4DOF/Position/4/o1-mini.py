import math

def inverse_kinematics(p: list) -> list[float]:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    x, y, z = p
    l0 = 0.4  # Link1: Base to Joint2 along Z-axis
    l1 = 0.4  # Link2: Joint2 to Joint3 along X-axis
    l2 = 0.4  # Link3: Joint3 to Joint4 along Z-axis
    l3 = 0.4  # Link4: Joint4 to End Effector along Z-axis

    # Step 1: Adjust the target position to account for the End Effector's offset
    x_eff = x
    y_eff = y
    z_eff = z - l3  # Subtract the end effector's offset

    # Step 2: Calculate the distance from the base to the wrist position
    D = math.sqrt(x_eff**2 + y_eff**2 + z_eff**2)

    # Define the manipulator's maximum reach to the wrist
    max_reach = l0 + l1 + l2  # 1.2 meters
    epsilon = 1e-6  # Small tolerance for floating-point comparisons

    # Step 2: Check if the position is reachable
    if D > (max_reach + epsilon):
        raise ValueError("Target position is out of the manipulator's reachable workspace.")

    # Step 3: Compute theta1 (base rotation around Z-axis)
    theta1 = math.atan2(y_eff, x_eff)

    # Project the target onto the XY-plane to find the planar distance
    r = math.sqrt(x_eff**2 + y_eff**2)

    # Compute theta3 using the law of cosines
    cos_theta3 = (r**2 + z_eff**2 - l1**2 - l2**2) / (2 * l1 * l2)
    # Clamp cos_theta3 to the valid range [-1, 1] to avoid numerical issues
    cos_theta3 = max(min(cos_theta3, 1.0), -1.0)
    # Compute theta3 (elbow-down configuration)
    theta3 = math.acos(cos_theta3)

    # Compute theta2
    sin_theta3 = math.sin(theta3)
    k1 = l1 + l2 * cos_theta3
    k2 = l2 * sin_theta3
    theta2 = math.atan2(z_eff, r) - math.atan2(k2, k1)

    # Compute theta4 to maintain the End Effector's orientation
    # Since no specific orientation is required, set theta4 to zero
    theta4 = 0.0

    return [theta1, theta2, theta3, theta4]
