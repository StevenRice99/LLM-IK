import math

def inverse_kinematics(p: list) -> list:
    """
    Solve the inverse kinematics to reach the position in the reference frame of the chain's origin.
    :param p: The position for the end effector to reach in a Cartesian [x, y, z].
    :return: A list of the joint values in radians needed for the end effector to reach the position.
    """
    # Extract target coordinates
    x, y, z = p

    # Calculate the angle needed
    theta = math.atan2(-y, z)

    return [theta]
