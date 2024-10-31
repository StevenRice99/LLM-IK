import math

def inverse_kinematics(p: list) -> list[float]:
    """
    Solve the inverse kinematics to reach the position and orientation, both in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    x, y, z = p

    # Compute cos(theta2)
    cos_theta2 = (z - 0.4) / 0.4
    # Clamp cos_theta2 to the valid range [-1, 1]
    cos_theta2 = max(min(cos_theta2, 1.0), -1.0)

    # Possible theta2 values
    theta2_options = [math.acos(cos_theta2), -math.acos(cos_theta2)]

    for theta2 in theta2_options:
        sin_theta2 = math.sin(theta2)

        # Avoid division by zero
        if abs(sin_theta2) < 1e-6:
            # When sin(theta2) is zero, x and y should be zero
            if abs(x) < 1e-6 and abs(y) < 1e-6:
                theta1 = 0.0  # Any value is acceptable
                return [theta1, theta2]
            else:
                continue  # Invalid solution
        else:
            # Compute theta1
            theta1 = math.atan2(-x, y)
            # Recalculate x and y to verify
            x_calc = 0.4 * math.sin(theta1) * sin_theta2
            y_calc = -0.4 * math.cos(theta1) * sin_theta2

            # Check if calculated x and y match the given values
            if abs(x - x_calc) < 1e-6 and abs(y - y_calc) < 1e-6:
                return [theta1, theta2]

    # If no valid solution is found
    raise ValueError("No valid solution found for the given position.")
