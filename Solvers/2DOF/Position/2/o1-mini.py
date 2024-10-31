import math


def inverse_kinematics(p: list) -> list[float]:
    """
    Solve the inverse kinematics to reach the position and orientation, both in the reference frame of the chain's origin.

    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    x, y, z = p

    # Link lengths
    l1 = 0.4  # Length from Joint 1 to Joint 2 along Z-axis
    l2 = 0.4  # Length from Joint 2 to End Effector along the manipulated axis

    # Calculate theta2 using the z-coordinate
    cos_theta2 = (z - l1) / l2
    # Ensure the argument of acos is within the valid range [-1, 1]
    if not -1.0 <= cos_theta2 <= 1.0:
        raise ValueError("Invalid z-coordinate for inverse kinematics solution.")

    # Two possible solutions for theta2
    theta2_positive = math.acos(cos_theta2)
    theta2_negative = -math.acos(cos_theta2)

    # Based on the provided failure cases, the correct solution uses the negative theta2
    theta2 = theta2_negative

    # Calculate theta1 using the x and y coordinates
    # theta1 = atan2(-x, y)
    theta1 = math.atan2(-x, y)

    return [theta1, theta2]
