import math


def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    # Calculate the distance from the base to the end effector
    distance = math.sqrt(p[0] ** 2 + p[1] ** 2 + (p[2] - 0.4) ** 2)

    # Handle the case where the end effector is at the maximum extension of the joint
    if distance == 0.4:
        theta = 0
    else:
        # Calculate the joint angle (theta) using the DH parameters
        theta = math.atan2(p[1], p[0]) + math.acos((0.4 ** 2 + distance ** 2 - p[2] ** 2) / (2 * 0.4 * distance))

    # Calculate the joint value in radians
    joint_value = theta

    return [joint_value]
