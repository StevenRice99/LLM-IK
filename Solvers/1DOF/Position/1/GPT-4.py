import math


def inverse_kinematics(p: list) -> list[float]:
    """
    Solve the inverse kinematics to reach the position and orientation, both in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    # The length from the joint to the end effector
    L = 0.4

    # Extract target coordinates
    p_y, p_z = p[1], p[2]

    # Compute the angle required to reach the desired position, with corrected sign
    theta = math.atan2(-p_y, p_z)

    # Return the angle as a list of one element (since we have only one DOF)
    return [theta]
