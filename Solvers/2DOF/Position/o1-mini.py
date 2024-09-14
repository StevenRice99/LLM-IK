import math


def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.

    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values [theta1, theta2] in radians needed for the end effector to reach the position.
    :raises ValueError: If the position is not reachable.
    """
    x_d, y_d, z_d = p

    # Compute cos(theta2) from the z-component
    cos_theta2 = z_d / 0.4 - 1

    # Check if the desired z is within the reachable range
    if cos_theta2 < -1 or cos_theta2 > 1:
        raise ValueError(f"Position {p} is not reachable. z-component out of range.")

    # Compute theta2 using arccos
    theta2 = math.acos(cos_theta2)

    # Compute sin(theta2) ensuring numerical stability
    sin_theta2_sq = 1 - cos_theta2 ** 2
    sin_theta2 = math.sqrt(sin_theta2_sq) if sin_theta2_sq >= 0 else 0.0

    # Define a small threshold to handle numerical precision issues
    epsilon = 1e-6

    if sin_theta2 < epsilon:
        # When sin(theta2) is close to zero, theta1 is undefined.
        # Check if x_d and y_d are also close to zero.
        if math.hypot(x_d, y_d) > epsilon:
            raise ValueError(f"Position {p} is not reachable due to inconsistent x and y components.")
        # Set theta1 to zero as it doesn't affect the position
        theta1 = 0.0
    else:
        # Compute theta1 using atan2 for proper quadrant determination
        theta1 = math.atan2(x_d, -y_d)

    return [theta1, theta2]
